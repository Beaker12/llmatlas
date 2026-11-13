#!/usr/bin/env python3
"""Build the chapter 6 notebook (unexecuted) from the chapter text and the cell sources below.

Code that the chapter prints (count_params, the KV-cache function, the router, the frequency
helpers, grouped-query attention) is copied from the chapter's fences, and RMSNorm, SwiGLU,
PreLNBlock and apply_rope from chapter 1's, so the notebook runs exactly what the text shows.

Usage: python tools/build_architecture_notebook.py
Then execute: jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=120 chapters/06-architectures/examples/architecture_demo.ipynb
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CH06 = ROOT / "chapters" / "06-architectures" / "index.md"
CH01 = ROOT / "chapters" / "01-foundations" / "index.md"
OUT = ROOT / "chapters" / "06-architectures" / "examples" / "architecture_demo.ipynb"
FENCE = re.compile(r"```python\n([\s\S]*?)```")


def fence(path: Path, marker: str) -> str:
    for m in FENCE.finditer(path.read_text(encoding="utf-8")):
        if marker in m.group(1):
            return m.group(1).strip("\n")
    raise SystemExit(f"no python fence in {path.name} contains {marker!r}")


def function(path: Path, name: str) -> str:
    """One top-level function from a fence, up to the next blank line plus unindented text."""
    text = fence(path, f"def {name}")
    m = re.search(rf"^def {name}\b[\s\S]*?(?=\n\n\S|\Z)", text, re.M)
    return m.group(0).rstrip()


CELLS: list[tuple[str, str]] = []
TAGS: dict[int, list[str]] = {}  # cell index -> metadata tags; `figure:<slug>` marks a plot the chapter shows


def md(text: str, tags: tuple[str, ...] = ()) -> None:
    CELLS.append(("markdown", text.strip("\n")))
    if tags:
        TAGS[len(CELLS) - 1] = list(tags)


def code(text: str, tags: tuple[str, ...] = ()) -> None:
    CELLS.append(("code", text.strip("\n")))
    if tags:
        TAGS[len(CELLS) - 1] = list(tags)


md(r"""
# Modern Architectures Demo

**Purpose**: This notebook builds the pieces of Chapter 6 and checks each against an independent reference: the parameter count of twelve real models from their configuration files (eight downloaded, four transcribed from papers), a grouped-query attention layer, a KV cache, a simplified latent cache, rotary-frequency scaling, a mixture-of-experts router, and a seeded comparison of post-norm and pre-norm training. Every section ends in assertions, so a silent change to any formula fails the notebook.

**Data**: model configurations are read from the Hugging Face hub at pinned commits when a network is available, and from copies committed in `data/` otherwise; set `LLMATLAS_OFFLINE=1` to skip the network. Both sources are verified against the committed checksums, so the results do not depend on which one is used, and the configuration cell prints the same label either way (the source of each file is kept in `SOURCES`). The training text is Jane Austen's *Pride and Prejudice*, public domain, from Project Gutenberg.
""")

md("## Setup")
code(r'''
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
plt.rcParams["figure.dpi"] = 100

DATA = Path("data")
OFFLINE = os.environ.get("LLMATLAS_OFFLINE") == "1"
USER_AGENT = os.environ.get("LLMATLAS_USER_AGENT", "llmatlas-notebook/0.1")


@contextmanager
def timed(label):
    start = time.perf_counter()
    yield
    print(f"[{label}: {time.perf_counter() - start:.1f} s]")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()
''')

md(r"""
(nb-arch-configs)=
## Real Configurations and Exact Parameter Counts

The ungated models are read from their pinned commits (`data/pins.json`). Each file is checked against `data/checksums.json`, whether it came from the hub or from the committed copy. The gated models, Llama 3.1 in its smaller and larger size and the Gemma models, cannot be downloaded without an account; their architectures are transcribed from their papers and checked against the exact parameter totals the hub publishes for them, or, for the multimodal Gemma model, against its paper's table.
""")
code(r'''
PINS = json.loads((DATA / "pins.json").read_text())
CHECKSUMS = json.loads((DATA / "checksums.json").read_text())["sha256"]
live_ok = not OFFLINE


def load_config(repo):
    """The model's config.json: from the hub at the pinned commit if reachable, else committed."""
    global live_ok
    name = f"configs/{repo.replace('/', '__')}.json"
    committed = (DATA / name).read_bytes()
    assert sha256(committed) == CHECKSUMS[name], f"committed {name} does not match its checksum"
    if live_ok:
        url = f"https://huggingface.co/{repo}/resolve/{PINS[repo]}/config.json"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=10) as response:
                live = response.read()
            if sha256(live) == CHECKSUMS[name]:
                return json.loads(live), "hub, pinned commit"
            return json.loads(committed), "committed copy (hub bytes differ)"
        except (urllib.error.URLError, TimeoutError, OSError) as error:  # incl. HTTP 401/403/404/429
            live_ok = False  # do not wait for a timeout on every remaining model
    return json.loads(committed), "committed copy"


CONFIGS, SOURCES = {}, {}  # SOURCES records where each file came from; the printed label does not depend on it
for repo in PINS:
    CONFIGS[repo], SOURCES[repo] = load_config(repo)
assert set(SOURCES.values()) <= {"hub, pinned commit", "committed copy", "committed copy (hub bytes differ)"}
for repo in PINS:
    print(f"{repo:32} {CONFIGS[repo]['model_type']:12} checksum verified")
''')

md(r"""
The four gated models, transcribed from their papers, except Gemma 2's vocabulary, which is the released checkpoint's as the hub reports it, and Gemma 3's layer shapes, which come from Google's reference implementation (the paper gives the counts but not the shapes); Gemma 3 is the text decoder of a multimodal model. Gemma 2's vocabulary: the paper's Table 1 lists 256,128 entries, the released checkpoint has 256,000 rows, and the hub's total is reproduced with 256,000. Gemma 2's head dimension (256) is larger than $d/n_h$, and its embeddings are tied.
""", tags=("static-numbers",))  # literature constants: the paper's and the checkpoint's vocabulary and head sizes
code(r'''
PAPER_CONFIGS = {
    "meta-llama/Llama-3.1-8B": {
        "model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
        "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336,
        "vocab_size": 128256, "tie_word_embeddings": False},
    "meta-llama/Llama-3.1-70B": {
        "model_type": "llama", "hidden_size": 8192, "num_hidden_layers": 80,
        "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672,
        "vocab_size": 128256, "tie_word_embeddings": False},
    "google/gemma-2-9b": {
        "model_type": "gemma2", "hidden_size": 3584, "num_hidden_layers": 42,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
        "intermediate_size": 14336, "vocab_size": 256000, "tie_word_embeddings": True},  # vocabulary: hub, not paper
    # Gemma 3 12B, text decoder only: shapes from Google's reference code, vocabulary 262,144
    "google/gemma-3-12b-pt": {
        "model_type": "gemma3_text", "hidden_size": 3840, "num_hidden_layers": 48,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
        "intermediate_size": 15360, "vocab_size": 262144, "tie_word_embeddings": True},
}
TEXT_ONLY = {"google/gemma-3-12b-pt"}  # multimodal checkpoint: the hub total includes the vision encoder
ALL_CONFIGS = {**CONFIGS, **PAPER_CONFIGS}
''')

md("The parameter counter and the KV-cache function are the ones printed in the chapter, copied from its text.")
code(fence(CH06, "def count_params"))
code(fence(CH06, "def kv_cache_elements_per_token"))

md(r"""
The checks below each catch a different kind of mistake. First, against the checkpoint: the element counts were read from the tensor shapes in each safetensors header (`tools/fetch_arch_samples.py`), which does not depend on whether a model is stored in `float32`, `BF16` or `FP8`. Second, against the hub's published total for the gated models. Third, against each vendor's own published figure, in that vendor's convention for "active" parameters; the other convention must miss.
""")
code(r'''
CHECKPOINT = json.loads((DATA / "checkpoint_counts.json").read_text())
HUB_TOTALS = json.loads((DATA / "hub_totals.json").read_text())
COUNTS = {repo: count_params(cfg) for repo, cfg in ALL_CONFIGS.items()}

# 1. Exact agreement with the element counts of the stored tensors
for repo in PINS:
    counted = CHECKPOINT[repo]["elements_counted"]
    assert COUNTS[repo]["total"] == counted, (repo, COUNTS[repo]["total"], counted)
print("checkpoint element counts reproduced exactly for", len(PINS), "models")
v3 = CHECKPOINT["deepseek-ai/DeepSeek-V3"]
print(f"  DeepSeek-V3 also stores {v3['elements_mtp_layer']:,} multi-token-prediction and "
      f"{v3['elements_fp8_scales']:,} FP8-scale elements, counted separately")

# 2. Gated models: the hub's public metadata reports the exact total
exact = [repo for repo in PAPER_CONFIGS if repo not in TEXT_ONLY]
for repo in exact:
    assert COUNTS[repo]["total"] == HUB_TOTALS[repo]["elements_total"], repo
print("hub totals reproduced exactly for", ", ".join(exact))
# Gemma 3: the paper's Table 1 gives 10,759M non-embedding parameters and a 417M vision encoder
gemma3 = PAPER_CONFIGS["google/gemma-3-12b-pt"]
non_embedding = COUNTS["google/gemma-3-12b-pt"]["total"] - gemma3["vocab_size"] * gemma3["hidden_size"]
rest = HUB_TOTALS["google/gemma-3-12b-pt"]["elements_total"] - COUNTS["google/gemma-3-12b-pt"]["total"]
assert round(non_embedding / 1e6) == 10759 and 417e6 <= rest < 430e6, (non_embedding, rest)
print(f"Gemma 3 12B text decoder: {non_embedding / 1e6:,.0f}M non-embedding, as in the paper; "
      f"{rest / 1e6:.1f}M of the hub total is vision")

# 3. Published figures (billions): (total, tolerance, active, convention, tolerance)
# Tolerances are half a unit in the last published digit, widened where the figure is consistent
# with truncation (Qwen 7.6156 -> 7.61, DeepSeek-V2-Lite 2.451 -> 2.4) or with rounding up
# (Mistral 7.2417 -> 7.3). OLMo 2 7B has no published total more precise than its name.
PUBLISHED = {
    "Qwen/Qwen2.5-7B": (7.61, 0.01, None, None, None),
    "Qwen/Qwen3-8B-Base": (8.2, 0.05, None, None, None),
    "allenai/OLMoE-1B-7B-0924": (6.9, 0.05, 1.3, "with", 0.05),
    "deepseek-ai/DeepSeek-V2-Lite": (15.7, 0.05, 2.4, "without", 0.1),
    "deepseek-ai/DeepSeek-V3": (671, 0.5, 37, "without", 0.5),
    "mistralai/Mistral-7B-v0.1": (7.3, 0.1, None, None, None),
    "mistralai/Mixtral-8x7B-v0.1": (46.7, 0.05, 12.9, "with", 0.05),
}
for repo, (total, tol, active, convention, tol_active) in PUBLISHED.items():
    got = COUNTS[repo]
    assert abs(got["total"] / 1e9 - total) <= tol, (repo, got["total"] / 1e9, total)
    if active is not None:
        other = "without" if convention == "with" else "with"
        right = got[f"active_{convention}_embedding"] / 1e9
        wrong = got[f"active_{other}_embedding"] / 1e9
        assert abs(right - active) <= tol_active, (repo, convention, right, active)
        assert abs(wrong - active) > tol_active, (repo, other, wrong, active)  # other convention misses
        print(f"{repo:32} active {right:7.3f}B ({convention} embedding) vs published {active}B; "
              f"{other}: {wrong:.3f}B misses")

# Coverage: the set contains a dense, a sparse and a latent-attention model
kinds = {cfg["model_type"] for cfg in CONFIGS.values()}
assert {"qwen2", "mixtral", "deepseek_v2", "deepseek_v3"} <= kinds
''')

md(r"""
The derived columns used in the chapter: embedding share, feed-forward share (dense models), and the KV-cache bytes per token over all layers in `bf16`.
""")
code(r'''
rows = []
for repo, cfg in ALL_CONFIGS.items():
    n, hidden, layers, vocab = COUNTS[repo]["total"], cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    tied = bool(cfg.get("tie_word_embeddings"))
    embedding_share = vocab * hidden * (1 if tied else 2) / n
    dense = "intermediate_size" in cfg and not any(k in cfg for k in ("num_experts", "num_local_experts", "n_routed_experts"))
    ffn_share = 3 * hidden * cfg["intermediate_size"] * layers / n if dense else None
    kv_bytes = 2 * kv_cache_elements_per_token(cfg)
    rows.append((repo, n, embedding_share, ffn_share, kv_bytes))

print(f"{'model':32} {'params':>10} {'embed %':>8} {'ffn %':>6} {'KV B/token':>11}")
for repo, n, emb, ffn, kv in rows:
    ffn_text = f"{100 * ffn:6.1f}" if ffn is not None else "     -"
    print(f"{repo:32} {n / 1e9:9.2f}B {100 * emb:8.1f} {ffn_text} {kv:11,}")

shares = [ffn for *_, ffn, _ in rows if ffn is not None]
assert 0.59 < min(shares) < 0.60 and 0.79 < max(shares) < 0.81  # OLMo 2 7B ... Llama 3.1 70B
kv_bytes = {repo: kv for repo, *_, kv in rows}
assert kv_bytes["meta-llama/Llama-3.1-8B"] == 131072 and kv_bytes["deepseek-ai/DeepSeek-V3"] == 70272
assert kv_bytes["deepseek-ai/DeepSeek-V3"] == 61 * (512 + 64) * 2  # MLA: latent + rotary key per layer

def attention_family(cfg):
    if cfg["model_type"].startswith("deepseek"):
        return "MLA"
    n_heads = cfg["num_attention_heads"]
    return "MHA" if (cfg.get("num_key_value_heads") or n_heads) == n_heads else "GQA"


family_colors = {"MHA": "#c44e52", "GQA": "#4c72b0", "MLA": "#55a868"}
order = sorted(kv_bytes, key=kv_bytes.get)
kib = [kv_bytes[name] / 1024 for name in order]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh([name.split("/")[1] for name in order], kib,
        color=[family_colors[attention_family(ALL_CONFIGS[name])] for name in order])
for row, value in enumerate(kib):
    ax.text(value * 1.06, row, f"{value:.4g}", va="center", fontsize=9)
ax.set_xscale("log")
ax.set_xlim(10, 2000)
ax.set_xlabel("KV cache per token, all layers (KiB, bf16)")
ax.set_title("What one token costs in the KV cache")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in family_colors.values()], list(family_colors),
          loc="lower right", fontsize=9)
plt.tight_layout()
plt.show()
''', tags=("figure:kv-bytes",))

md(r"""
(nb-arch-block)=
## The Block: RMSNorm, Grouped-Query Attention, SwiGLU

`RMSNorm`, `SwiGLU`, `PreLNBlock` and `apply_rope` below are copied from the foundations chapter ({ref}`sec-ch1-modern-block`, {ref}`sec-ch1-rope`). The new piece is the attention layer, which supports any number of key/value heads (`n_kv_heads = n_heads` is multi-head attention, `1` is multi-query) and exposes `project` and `attend` separately so the cache section can reuse them.
""")
code(fence(CH01, "class RMSNorm"))
code(function(CH01, "apply_rope"))
code(r'''
class GQAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, rope_base=10000.0):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv, self.d_head = n_heads, n_kv_heads, d_model // n_heads
        self.rope_base = rope_base
        self.wq = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.wo = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def project(self, x, positions):
        """Queries, keys (after RoPE) and values: [batch, heads, seq, d_head]."""
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        return apply_rope(q, positions, self.rope_base), apply_rope(k, positions, self.rope_base), v

    def attend(self, q, k, v, causal=True):
        group = self.n_heads // self.n_kv
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    def forward(self, x, _keys=None, _values=None):  # the signature PreLNBlock calls
        b, t, _ = x.shape
        q, k, v = self.project(x, torch.arange(t))
        return self.wo(self.attend(q, k, v).transpose(1, 2).reshape(b, t, -1))


class PostLNBlock(nn.Module):
    """The 2017 arrangement with RMSNorm and SwiGLU: normalize after each residual addition."""

    def __init__(self, d_model, attention):
        super().__init__()
        self.attention = attention
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attention(x, x, x))
        return self.norm2(x + self.ffn(x))


class TinyLM(nn.Module):
    """A byte-level decoder with tied embeddings; `style` is "pre" or "post"."""

    def __init__(self, vocab=256, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, style="pre"):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        block = PreLNBlock if style == "pre" else PostLNBlock
        self.blocks = nn.ModuleList(
            [block(d_model, GQAttention(d_model, n_heads, n_kv_heads)) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model) if style == "pre" else nn.Identity()

    def forward(self, tokens):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x) @ self.embed.weight.T
''')

md("Checks. RMSNorm against PyTorch's `nn.RMSNorm`; grouped-query attention (and multi-query, the one-group case) against a plain softmax implementation that uses multi-head attention with the key/value weights repeated per group; causality; and the module's parameter count against `count_params`.")
code(r'''
torch.manual_seed(1)

# RMSNorm vs torch.nn.RMSNorm, float64, random weights
ours, ref = RMSNorm(64).double(), nn.RMSNorm(64, eps=1e-6).double()
with torch.no_grad():
    ours.weight.copy_(torch.randn(64)); ref.weight.copy_(ours.weight)
x = torch.randn(3, 5, 64, dtype=torch.float64)
assert torch.allclose(ours(x), ref(x), atol=1e-12)


def reference_attention(att, x):
    """Explicit softmax attention with one key/value head per query head (weights repeated)."""
    b, t, _ = x.shape
    h, g = att.n_heads, att.n_heads // att.n_kv
    def expand(weight):  # [n_kv * d_head, d_model] -> [n_heads * d_head, d_model]
        w = weight.view(att.n_kv, att.d_head, -1)
        return w.repeat_interleave(g, dim=0).reshape(h * att.d_head, -1)
    q = (x @ att.wq.weight.T).view(b, t, h, att.d_head).transpose(1, 2)
    k = (x @ expand(att.wk.weight).T).view(b, t, h, att.d_head).transpose(1, 2)
    v = (x @ expand(att.wv.weight).T).view(b, t, h, att.d_head).transpose(1, 2)
    pos = torch.arange(t)
    q, k = apply_rope(q, pos), apply_rope(k, pos)
    scores = q @ k.transpose(-1, -2) / math.sqrt(att.d_head)
    scores = scores.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), float("-inf"))
    out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
    return out @ att.wo.weight.T


x = torch.randn(2, 9, 64, dtype=torch.float64)
for n_kv in (4, 2, 1):  # multi-head, grouped-query (2 groups), multi-query
    att = GQAttention(64, 4, n_kv).double()
    assert torch.allclose(att(x), reference_attention(att, x), atol=1e-10), n_kv
print("RMSNorm and attention match their references for 4, 2 and 1 key/value heads")

# The chapter's grouped_query_attention function is the same computation as the module's attend
''')
code(fence(CH06, "def grouped_query_attention"))
code(r'''
att = GQAttention(64, 4, 2).double()
q, k, v = att.project(x, torch.arange(9))
assert torch.allclose(grouped_query_attention(q, k, v), att.attend(q, k, v), atol=1e-12)

# Causality: changing the last tokens does not change earlier outputs
model = TinyLM().double()
tokens = torch.randint(0, 256, (2, 16))
changed = tokens.clone()
changed[:, 10:] = torch.randint(0, 256, (2, 6))
with torch.no_grad():
    a, b = model(tokens), model(changed)
assert torch.allclose(a[:, :10], b[:, :10], atol=1e-12) and not torch.allclose(a[:, 10:], b[:, 10:])

# The module has exactly the parameters count_params predicts for its configuration
TINY = {"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
        "num_key_value_heads": 2, "intermediate_size": 170, "vocab_size": 256,
        "tie_word_embeddings": True}
measured = sum(p.numel() for p in TinyLM().parameters())
assert measured == count_params(TINY)["total"], (measured, count_params(TINY)["total"])
print(f"TinyLM has {measured:,} parameters, as count_params predicts")
''')

md(r"""
(nb-arch-kvcache)=
## The KV Cache

Decoding one token at a time with a cache must give the same logits as running the whole sequence at once. The loop below is the cache in its simplest form: every layer appends the new token's keys and values, and the query attends to everything stored (no mask is needed, because the cache holds only the past).
""")
code(r'''
@torch.no_grad()
def decode_with_cache(model, tokens):
    caches = [{"k": [], "v": []} for _ in model.blocks]
    logits = []
    for pos in range(tokens.shape[1]):
        x = model.embed(tokens[:, pos:pos + 1])
        for block, cache in zip(model.blocks, caches):
            att = block.attention
            q, k, v = att.project(block.norm1(x), torch.tensor([pos]))
            cache["k"].append(k)
            cache["v"].append(v)
            keys, values = torch.cat(cache["k"], dim=2), torch.cat(cache["v"], dim=2)
            out = att.attend(q, keys, values, causal=False).transpose(1, 2).reshape(x.shape[0], 1, -1)
            x = x + att.wo(out)
            x = x + block.ffn(block.norm2(x))
        logits.append(model.final_norm(x) @ model.embed.weight.T)
    return torch.cat(logits, dim=1), caches


tokens = torch.randint(0, 256, (2, 24))
for n_kv in (4, 2, 1):
    model = TinyLM(n_kv_heads=n_kv).double()
    with torch.no_grad():
        full = model(tokens)
    cached, caches = decode_with_cache(model, tokens)
    assert torch.allclose(full, cached, atol=1e-10), n_kv
    # Measured cache size per token equals the chapter's formula
    stored = sum(t.numel() for c in caches for t in c["k"] + c["v"]) / (tokens.shape[0] * tokens.shape[1])
    cfg = {**TINY, "num_key_value_heads": n_kv}
    assert stored == kv_cache_elements_per_token(cfg), (stored, kv_cache_elements_per_token(cfg))
    print(f"n_kv={n_kv}: cached decoding equals the full pass; cache {int(stored)} elements/token "
          f"(multi-head would be {kv_cache_elements_per_token({**TINY, 'num_key_value_heads': 4})})")
''')

md(r"""
A simplified multi-head latent attention layer. Only a small latent $c_i$ and a shared rotary key are cached. The reference expands the latent into full keys and values and runs ordinary attention; the decoding path never builds them, because the key up-projection is absorbed into the query and the value up-projection is applied after the weighted sum. This is a simplified layer (no query compression, one rotary key), not DeepSeek's implementation.
""")
code(r'''
class MiniMLA(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_nope=16, d_rope=8, d_v=16, d_c=16, base=10000.0):
        super().__init__()
        self.h, self.dn, self.dr, self.dv, self.dc, self.base = n_heads, d_nope, d_rope, d_v, d_c, base
        self.wq = nn.Linear(d_model, n_heads * (d_nope + d_rope), bias=False)
        self.w_dkv = nn.Linear(d_model, d_c, bias=False)       # down-projection: what is cached
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)     # rotary key, shared by all heads
        self.w_uk = nn.Linear(d_c, n_heads * d_nope, bias=False)
        self.w_uv = nn.Linear(d_c, n_heads * d_v, bias=False)
        self.wo = nn.Linear(n_heads * d_v, d_model, bias=False)

    def queries(self, x, positions):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.h, self.dn + self.dr).transpose(1, 2)
        return q[..., :self.dn], apply_rope(q[..., self.dn:], positions, self.base)

    def rotary_key(self, x, positions):
        return apply_rope(self.w_kr(x)[:, None], positions, self.base)[:, 0]  # [batch, seq, d_rope]

    def forward_expanded(self, x):
        """Reference: expand the latent to full keys and values, then ordinary causal attention."""
        b, t, _ = x.shape
        pos = torch.arange(t)
        q_nope, q_rope = self.queries(x, pos)
        c = self.w_dkv(x)
        k_nope = self.w_uk(c).view(b, t, self.h, self.dn).transpose(1, 2)
        v = self.w_uv(c).view(b, t, self.h, self.dv).transpose(1, 2)
        k_rope = self.rotary_key(x, pos)[:, None].expand(-1, self.h, -1, -1)
        out = F.scaled_dot_product_attention(
            torch.cat([q_nope, q_rope], -1), torch.cat([k_nope, k_rope], -1), v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))

    def decode_step(self, x_t, cache, position):
        """One token using only the cache of latents and rotary keys."""
        pos = torch.tensor([position])
        q_nope, q_rope = self.queries(x_t, pos)
        cache["c"].append(self.w_dkv(x_t))
        cache["kr"].append(self.rotary_key(x_t, pos))
        c, kr = torch.cat(cache["c"], 1), torch.cat(cache["kr"], 1)
        w_uk = self.w_uk.weight.view(self.h, self.dn, self.dc)
        w_uv = self.w_uv.weight.view(self.h, self.dv, self.dc)
        q_latent = torch.einsum("bhtn,hnc->bhtc", q_nope, w_uk)            # absorb W_UK into the query
        scores = torch.einsum("bhtc,bsc->bhts", q_latent, c) + torch.einsum("bhtr,bsr->bhts", q_rope, kr)
        weights = (scores / math.sqrt(self.dn + self.dr)).softmax(-1)
        latent = torch.einsum("bhts,bsc->bhtc", weights, c)                  # weighted sum of latents
        out = torch.einsum("bhtc,hvc->bhtv", latent, w_uv)                   # value up-projection after
        return self.wo(out.transpose(1, 2).reshape(x_t.shape[0], 1, -1))


mla = MiniMLA().double()
x = torch.randn(2, 12, 64, dtype=torch.float64)
with torch.no_grad():
    expanded = mla.forward_expanded(x)
    cache = {"c": [], "kr": []}
    stepped = torch.cat([mla.decode_step(x[:, i:i + 1], cache, i) for i in range(12)], dim=1)
assert torch.allclose(expanded, stepped, atol=1e-10)

per_token = mla.dc + mla.dr
full = mla.h * (mla.dn + mla.dr + mla.dv)  # per token: every head's key (nope + rope) and value
assert sum(t.numel() for t in cache["c"] + cache["kr"]) / (2 * 12) == per_token
print(f"latent decoding equals the expanded attention; cache {per_token} elements/token "
      f"versus {full} for full per-head keys and values ({full / per_token:.1f}x)")
''')

md(r"""
(nb-arch-rope)=
## Rotary Positions and Longer Contexts

First the defining properties of RoPE, for the function copied from the foundations chapter ({ref}`sec-ch1-rope`): rotation preserves norms, and a query-key product depends only on the offset. Then the frequency arithmetic of the chapter, in `float64`: the NTK-aware base $b' = b\, s_{ctx}^{d_k/(d_k-2)}$ leaves the highest frequency alone and divides the lowest by exactly the scale $s_{ctx}$ (the variable `s` below).
""")
code(r'''
torch.manual_seed(2)
q, k = torch.randn(1, 64), torch.randn(1, 64)
rotated = apply_rope(q.expand(10, 64), torch.arange(10))
assert torch.allclose(rotated.norm(dim=-1), q.norm().expand(10), rtol=1e-5)  # norms preserved


def score(m, n):
    return (apply_rope(q, torch.tensor([m])) @ apply_rope(k, torch.tensor([n])).T).item()


for shift in (1, 7, 100):
    assert abs(score(20, 13) - score(20 + shift, 13 + shift)) < 1e-3  # depends on m - n only
print("RoPE preserves norms and depends only on the offset")
''')
code(fence(CH06, "def rope_frequencies"))
code(r'''
D, BASE, TRAINED = 64, 10000.0, 4096
theta = rope_frequencies(D, BASE)
last = D // 2 - 1

for s in (2.0, 4.0, 8.0, 40.0):
    scaled = rope_frequencies(D, ntk_aware_base(BASE, s, D))
    assert abs(scaled[0] - theta[0]) < 1e-12                       # highest frequency unchanged
    assert abs(theta[last] / scaled[last] - s) < 1e-9 * s          # lowest divided by exactly s

# Position interpolation, s = 4: trained on positions 0 .. TRAINED-1, extended to 0 .. TRAINED*s-1.
s = 4.0
low = [i for i in range(D // 2) if TRAINED * theta[i] < 2 * math.pi]   # turn less than once in training
high = [i for i in range(D // 2) if TRAINED * theta[i] >= 2 * math.pi]
seen = (TRAINED - 1) * theta[low]                                      # largest angle seen in training
interpolated = (TRAINED - 1 / s) * theta[low]                          # largest angle after dividing positions by s
extrapolated = (TRAINED * s - 1) * theta[low]                          # largest angle when positions are not rescaled
assert len(low) > 0 and last in low
assert torch.all(interpolated > seen)                  # the last rescaled position, 4095.75, is past 4095 ...
assert torch.all(interpolated < seen + theta[low])     # ... but within one position's rotation of it
assert torch.all(extrapolated > seen + theta[low])     # without rescaling, angles run far past it
assert high[0] == 0 and all(TRAINED * theta[i] > 2 * math.pi for i in high)  # these wrap many times already
print(f"{len(low)} of {D // 2} frequency pairs turn less than once in {TRAINED} positions; "
      f"interpolation keeps them within one position's rotation of the trained range, extrapolation does not")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
index = np.arange(D // 2)
ax.semilogy(index, theta.numpy(), label="original (base 10,000)")
ax.semilogy(index, (theta / s).numpy(), "--", label="position interpolation (all / 4)")
ax.semilogy(index, rope_frequencies(D, ntk_aware_base(BASE, s, D)).numpy(), label="NTK-aware (base x 4^(d/(d-2)))")
ax.set_xlabel("frequency pair index i")
ax.set_ylabel("angle per position (rad)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
''')

md(r"""
(nb-arch-moe)=
## A Mixture-of-Experts Router

The router from the chapter, run on a batch of random tokens; the token count `T`, the number of experts `E` and the number each token is routed to, `K`, are set in the cell below. Each row of the router weight has unit norm, so no expert is favored at initialization, and the dispatch fraction of an expert is its share of the routed slots (they sum to one).
""")
code(fence(CH06, "def route_top_k"))
code(r'''
E, K, T, D_MODEL = 8, 2, 4096, 64


def make_router(seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(E, D_MODEL, generator=g)
    return w / w.norm(dim=-1, keepdim=True), torch.randn(T, D_MODEL, generator=g)


# Spread of the dispatch fractions over 20 seeds, which sets the tolerance used below
deviations = []
for seed in range(20):
    w, x = make_router(seed)
    ids, _, _ = route_top_k(x, w, K)
    f = torch.bincount(ids.flatten(), minlength=E).float() / (T * K)
    deviations.append((f - 1 / E).abs().max().item())
TOL = 0.03
print(f"largest deviation of any expert's share from 1/{E} over 20 seeds: {max(deviations):.4f} (tolerance {TOL})")
assert max(deviations) < TOL

w, x = make_router(0)
ids, gates, aux_balanced = route_top_k(x, w, K)
assert torch.allclose(gates.sum(-1), torch.ones(T)) and ids.shape == (T, K)

# A router that favors expert 0: shift every token towards its weight row
ids_b, _, aux_biased = route_top_k(x + 3 * w[0], w, K)
print(f"load-balancing loss: balanced {aux_balanced:.3f}, biased {aux_biased:.3f} (uniform routing gives 1)")
assert 0.95 < aux_balanced < 1.1 and aux_biased > 2

# Capacity factor 1.0: each expert takes at most ceil(c * k * T / E) slots, in token order
CAPACITY = math.ceil(1.0 * K * T / E)


def dropped_vectorised(ids):
    flat = ids.flatten()                                            # slot order: token-major
    onehot = F.one_hot(flat, E)
    position = (onehot.cumsum(0) - 1).gather(1, flat[:, None])[:, 0]   # rank within its expert
    return int((position >= CAPACITY).sum())


def dropped_loop(ids):
    taken, dropped = [0] * E, 0
    for expert in ids.flatten().tolist():
        if taken[expert] >= CAPACITY:
            dropped += 1
        else:
            taken[expert] += 1
    return dropped


for name, routed in (("balanced", ids), ("biased", ids_b)):
    assert dropped_vectorised(routed) == dropped_loop(routed)
    print(f"{name:9} router drops {dropped_loop(routed):5d} of {T * K} slots at capacity factor 1.0")
assert dropped_loop(ids_b) > 5 * max(dropped_loop(ids), 1)

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for ax, (name, routed) in zip(axes, (("balanced", ids), ("biased toward expert 0", ids_b))):
    ax.bar(range(E), torch.bincount(routed.flatten(), minlength=E).numpy() / (T * K))
    ax.axhline(1 / E, color="k", lw=0.8, ls="--")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("expert")
axes[0].set_ylabel("share of routed slots")
plt.tight_layout()
plt.show()
''')

md(r"""
A full layer, checked against a per-token reference that follows Mixtral's formulation literally: take the router's largest *logits*, as many as the layer routes each token to, then apply a softmax over just those. The chapter's router renormalizes the full softmax over the selected experts, which is the same thing. The layer is then placed in a small model, and its total and active parameters are compared with `count_params`.
""")
code(r'''
class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, top_k, hidden):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, hidden) for _ in range(n_experts)])

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        top_p, top_i = self.router(flat).softmax(-1).topk(self.top_k, dim=-1)
        gates = top_p / top_p.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token, slot = (top_i == e).nonzero(as_tuple=True)
            if token.numel():
                out.index_add_(0, token, gates[token, slot].unsqueeze(-1) * expert(flat[token]))
        return out.view(x.shape)


def moe_reference(layer, x):
    flat, rows = x.reshape(-1, x.shape[-1]), []
    for t in range(flat.shape[0]):
        logits, chosen = layer.router(flat[t]).topk(layer.top_k)
        g = logits.softmax(-1)                                       # softmax over the selected logits
        rows.append(sum(g[j] * layer.experts[int(chosen[j])](flat[t]) for j in range(layer.top_k)))
    return torch.stack(rows).view(x.shape)


layer = MoELayer(64, 8, 2, 96).double()
x = torch.randn(2, 6, 64, dtype=torch.float64)
with torch.no_grad():
    assert torch.allclose(layer(x), moe_reference(layer, x), atol=1e-10)

# One expert with one slot is just the dense feed-forward network
one = MoELayer(64, 1, 1, 96).double()
with torch.no_grad():
    assert torch.allclose(one(x), one.experts[0](x), atol=1e-12)

# Total and active parameters of a small sparse model
moe_model = TinyLM(n_kv_heads=2)
for block in moe_model.blocks:
    block.ffn = MoELayer(64, 8, 2, 96)
cfg = {"model_type": "mixtral", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
       "num_key_value_heads": 2, "intermediate_size": 96, "num_local_experts": 8,
       "num_experts_per_tok": 2, "vocab_size": 256, "tie_word_embeddings": True}
counts = count_params(cfg)
total = sum(p.numel() for p in moe_model.parameters())
idle = 4 * (8 - 2) * sum(p.numel() for p in moe_model.blocks[0].ffn.experts[0].parameters())
assert total == counts["total"] and total - idle == counts["active_with_embedding"]
print(f"sparse toy model: {total:,} total parameters, {total - idle:,} active per token "
      f"({100 * (total - idle) / total:.0f}%)")
''')

CLAIMS_CELL = r"""
import numpy as np

STUCK, TRAINS = 3.0, 2.5  # nats: a byte-frequency model sits at 3.11; below 2.5 a model is using context


def norm_claims(arms):
    # each statement this section makes about the sweep, as a boolean, from its arms (lr, warmup, pre, post)
    by = {(a["lr"], a["warmup"]): {k: np.array(a[k]) for k in ("pre", "post")} for a in arms}
    top, warm = by[(3e-3, 0)], by[(3e-3, 100)]["post"]
    lower = [key for key in by if key[0] < 3e-3]
    return {
        "at 3e-3 without warm-up, post-norm stays at the byte-frequency loss and pre-norm trains, in every seed":
            bool((top["post"] > STUCK).all() and (top["pre"] < TRAINS).all()),
        "at 3e-3 with warm-up, post-norm trains in some seeds but not in all":
            bool((warm < TRAINS).any() and (warm > STUCK).any()),
        "at 1e-3 and 3e-4, with and without warm-up, post-norm ends lower than pre-norm in every seed":
            bool(all((by[key]["post"] < by[key]["pre"]).all() for key in lower)),
    }
""".strip("\n")


md(r"""
(nb-arch-train)=
## Post-Norm Against Pre-Norm

A six-layer byte-level model is trained on the first 150 KB of *Pride and Prejudice*, once with each norm placement, for 400 steps of 8 sequences of 64 bytes. A model that has only learned byte frequencies sits near the entropy of the text's byte distribution, which the next cell computes with the model's size; anything clearly lower means it is using context. For each of five seeds both variants see the same batches. The score is the mean training loss over the last 20 steps. Everything is single-threaded and deterministic, so a seed reproduces exactly.

The comparison is a sweep over three learning rates ($3 \times 10^{-3}$, $10^{-3}$, $3 \times 10^{-4}$), each with no warm-up and with a linear warm-up over the first 100 steps: six arms of ten runs. **Live:** the $3 \times 10^{-3}$, no-warm-up arm is trained below. **Loaded:** the other five arms are read from `data/norm_sweep.json`, which `tools/build_architecture_notebook.py --sweep` wrote by running this notebook's own training cell; the file records that cell's SHA-256 and the torch version, and `--sweep --check` recomputes every arm.
""", tags=("static-numbers",))  # experiment settings: model size, data, steps, rates, seeds
code(r'''
import numpy as np

_counts = np.bincount(np.frombuffer((DATA / "pride_and_prejudice_150k.txt").read_bytes(), dtype=np.uint8), minlength=256)
_freq = _counts[_counts > 0] / _counts.sum()
BYTE_FREQ_LOSS = float(-(_freq * np.log(_freq)).sum())
n_params = sum(p.numel() for p in TinyLM(n_layers=6, n_kv_heads=4).parameters())
print(f"model: {n_params:,} parameters; byte-frequency loss of the text: {BYTE_FREQ_LOSS:.2f} nats")
assert n_params == 311_360 and round(BYTE_FREQ_LOSS, 2) == 3.11
''')
TRAIN_CELL = r'''
raw = (DATA / "pride_and_prejudice_150k.txt").read_bytes()
assert sha256(raw) == CHECKSUMS["pride_and_prejudice_150k.txt"]
TEXT = torch.tensor(list(raw), dtype=torch.long)
BATCH, SEQ, STEPS, LR = 8, 64, 400, 3e-3


def train(style, seed, lr=LR, warmup=0):
    """Training losses of one run; `warmup` > 0 raises the rate linearly over that many steps."""
    torch.manual_seed(seed)
    model = TinyLM(n_layers=6, n_kv_heads=4, style=style)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if warmup:
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: min(1.0, (t + 1) / warmup))
    g = torch.Generator().manual_seed(1000 + seed)           # the same batches for both variants
    losses = []
    for _ in range(STEPS):
        start = torch.randint(0, len(TEXT) - SEQ - 1, (BATCH,), generator=g)
        index = start[:, None] + torch.arange(SEQ + 1)
        batch = TEXT[index]
        loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, 256), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if warmup:
            schedule.step()
        losses.append(loss.item())
    return losses


def score(losses):
    """The score of a run: the mean training loss over its last 20 steps."""
    return float(np.mean(losses[-20:]))


print(f"{sum(p.numel() for p in TinyLM(n_layers=6, n_kv_heads=4).parameters()):,} parameters; "
      f"{BATCH * SEQ:,} tokens per step; uniform-distribution loss is ln 256 = {math.log(256):.3f}")
RESULTS = {"pre": [], "post": []}
'''
code(TRAIN_CELL)
md("One cell per run keeps each well inside the time limit of a notebook cell on a small machine; each cell prints the time its run took.")
for _style in ("pre", "post"):
    for _seed in range(5):
        code(f'''
with timed("{_style}-norm, seed {_seed}"):
    RESULTS["{_style}"].append(train("{_style}", {_seed}))
print("final loss (mean of last 20 steps):", round(float(np.mean(RESULTS["{_style}"][-1][-20:])), 3))
''')
code(r'''
final = {style: np.array([np.mean(r[-20:]) for r in runs]) for style, runs in RESULTS.items()}
gap = final["post"] - final["pre"]
spread = max(final["pre"].std(), final["post"].std())
print(f"pre  {final['pre'].mean():.3f} +- {final['pre'].std():.3f}")
print(f"post {final['post'].mean():.3f} +- {final['post'].std():.3f}")
print(f"gap per seed {np.round(gap, 3)}; mean gap {gap.mean():.3f} = {gap.mean() / spread:.1f}x the larger spread")

assert all(np.isfinite(v).all() for v in final.values())

# Reproducibility: re-running one seed gives the identical curve
assert train("pre", 0) == RESULTS["pre"][0]

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for style, color in (("pre", "#4c72b0"), ("post", "#c44e52")):
    curves = np.array(RESULTS[style])
    ax.plot(curves.mean(0), color=color, label=f"{style}-norm (mean of {len(curves)} seeds)")
    ax.fill_between(range(STEPS), curves.min(0), curves.max(0), color=color, alpha=0.2)
ax.set_xlabel("step")
ax.set_ylabel("training loss (nats per byte)")
ax.legend()
plt.tight_layout()
plt.show()
''')

md(r"""
### The sweep: one live arm, five loaded

The row marked *live* is the runs above; the other rows are *loaded* from `data/norm_sweep.json`. Each statement printed after the table is asserted on these numbers.
""")
code(CLAIMS_CELL)
code(r'''
SWEEP = json.loads((DATA / "norm_sweep.json").read_text())
recorded = next(a for a in SWEEP["arms"] if (a["lr"], a["warmup"]) == (LR, 0))
difference = max(float(np.abs(final[k] - np.array(recorded[k])).max()) for k in ("pre", "post"))
if SWEEP["torch"] == torch.__version__:  # same version: the live arm reproduces the recorded one exactly
    assert difference == 0.0
print(f"sweep recorded with torch {SWEEP['torch']}, this kernel {torch.__version__}; "
      f"largest live-recorded difference {difference:.1g}")
ARMS = [{**a, **{k: [float(v) for v in final[k]] for k in ("pre", "post")}} if (a["lr"], a["warmup"]) == (LR, 0)
        else a for a in SWEEP["arms"]]

print(f"{'rate':>6} {'warm-up':>8} {'source':>7} {'pre-norm':>9} {'post-norm':>10}  (mean over 5 seeds)")
for a in ARMS:
    source = "live" if (a["lr"], a["warmup"]) == (LR, 0) else "loaded"
    print(f"{a['lr']:6.0e} {a['warmup']:8d} {source:>7} {np.mean(a['pre']):9.3f} {np.mean(a['post']):10.3f}")
for claim, holds in norm_claims(ARMS).items():
    print(f"{claim}: {holds}")
    assert holds, claim

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for style, color in (("pre", "#4c72b0"), ("post", "#c44e52")):
    for warmup, line in ((0, "-"), (100, "--")):
        rows = [a for a in ARMS if a["warmup"] == warmup]
        rates = [a["lr"] for a in rows]
        mean = np.array([np.mean(a[style]) for a in rows])
        low = mean - np.array([np.min(a[style]) for a in rows])
        high = np.array([np.max(a[style]) for a in rows]) - mean
        ax.errorbar(rates, mean, yerr=[low, high], fmt="o" + line, color=color, capsize=3,
                    label=f"{style}-norm, {'no warm-up' if warmup == 0 else f'{warmup}-step warm-up'}")
ax.axhline(BYTE_FREQ_LOSS, color="gray", lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel("peak learning rate")
ax.set_ylabel("final loss (nats per byte; mean, min-max over 5 seeds)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
''')

md(r"""
## Summary

The summary below is printed from the results computed above; each line is asserted.
""")
code(r'''
assert all(COUNTS[repo]["total"] == CHECKPOINT[repo]["elements_counted"] for repo in PINS)
assert all(COUNTS[repo]["total"] == HUB_TOTALS[repo]["elements_total"] for repo in exact)
print(f"checkpoint element counts reproduced exactly for {len(PINS)} models, hub totals for {len(exact)} gated models;")
print(f"  Gemma 3 12B text decoder: {non_embedding / 1e6:,.0f}M non-embedding parameters, as in the paper")
smallest, largest = min(kv_bytes, key=kv_bytes.get), max(kv_bytes, key=kv_bytes.get)
ratio = kv_bytes["allenai/OLMo-2-1124-7B"] / kv_bytes["meta-llama/Llama-3.1-8B"]
assert ratio == 4 and kv_bytes[smallest] < kv_bytes["deepseek-ai/DeepSeek-V3"] < kv_bytes[largest]
print(f"KV cache per token in bf16: {kv_bytes[smallest]:,} bytes ({smallest}) to {kv_bytes[largest]:,} ({largest});"
      f" OLMo 2 7B's multi-head cache is {ratio:.0f} times Llama 3.1 8B's")
assert per_token < full
print(f"latent cache: {per_token} numbers per token per layer against {full} for the expanded heads")
assert 0.95 < aux_balanced < 1.1 < 2 < aux_biased
print(f"router: load-balancing loss {aux_balanced:.3f} balanced, {aux_biased:.3f} biased toward one expert")
for claim, holds in norm_claims(ARMS).items():
    assert holds, claim
    print(f"norm placement: {claim}")
''')



def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for index, (kind, text) in enumerate(CELLS):
        cell = nbf.v4.new_markdown_cell(text) if kind == "markdown" else nbf.v4.new_code_cell(text)
        if index in TAGS:
            cell.metadata["tags"] = TAGS[index]
        nb.cells.append(cell)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} with {len(nb.cells)} cells")


# --- the learning-rate sweep behind Section 7 (decision D22) -------------------------------------------------
# `--sweep` runs every arm (each norm placement, five seeds, three rates, with and without warm-up) with the
# notebook's own training cell and writes the scores to data/norm_sweep.json; `--sweep --check` recomputes every
# arm and exits non-zero unless each statement the notebook asserts from the sweep still holds. Runs are
# independent, seeded and single-threaded, so running them in several processes gives the same numbers as a
# serial run.
SWEEP_JSON = OUT.parent / "data" / "norm_sweep.json"
SWEEP_RATES = (3e-3, 1e-3, 3e-4)
SWEEP_WARMUPS = (0, 100)
SWEEP_SEEDS = tuple(range(5))
_SWEEP_NS: dict = {}


def train_cell_sha256() -> str:
    """SHA-256 of the training cell's source exactly as it appears in the notebook."""
    import hashlib
    return hashlib.sha256(TRAIN_CELL.strip("\n").encode("utf-8")).hexdigest()


def _sweep_worker_init() -> None:
    """Run the notebook's code cells up to and including the training cell, offline and without output."""
    import contextlib
    import io
    import os
    os.environ["LLMATLAS_OFFLINE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(OUT.parent)
    stop = CELLS.index(("code", TRAIN_CELL.strip("\n")))
    with contextlib.redirect_stdout(io.StringIO()):
        for kind, text in CELLS[:stop + 1]:
            if kind == "code":
                exec(compile(text, "<notebook cell>", "exec"), _SWEEP_NS)


def _sweep_run(job):
    style, seed, lr, warmup = job
    return job, _SWEEP_NS["score"](_SWEEP_NS["train"](style, seed, lr=lr, warmup=warmup))


def run_sweep(processes: int) -> dict:
    import multiprocessing
    jobs = [(style, seed, lr, warmup) for lr in SWEEP_RATES for warmup in SWEEP_WARMUPS
            for style in ("pre", "post") for seed in SWEEP_SEEDS]
    with multiprocessing.get_context("spawn").Pool(processes, initializer=_sweep_worker_init) as pool:
        scores = dict(pool.map(_sweep_run, jobs, chunksize=1))
    import torch
    return {
        "description": "Final training loss (mean of the last 20 of 400 steps) of the Section 7 model, per arm "
                       "and seed; written by tools/build_architecture_notebook.py --sweep.",
        "torch": torch.__version__,
        "train_cell_sha256": train_cell_sha256(),
        "seeds": list(SWEEP_SEEDS),
        "arms": [{"lr": lr, "warmup": warmup,
                  **{style: [scores[(style, seed, lr, warmup)] for seed in SWEEP_SEEDS] for style in ("pre", "post")}}
                 for lr in SWEEP_RATES for warmup in SWEEP_WARMUPS],
    }


def sweep_claims(sweep: dict) -> dict:
    """Each statement the notebook asserts from the sweep, evaluated on `sweep` (the notebook's own cell)."""
    namespace: dict = {}
    exec(compile(CLAIMS_CELL, "<claims cell>", "exec"), namespace)
    return namespace["norm_claims"](sweep["arms"])


def main(argv=None) -> int:
    import argparse
    import json
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="run the Section 7 learning-rate sweep")
    parser.add_argument("--check", action="store_true", help="with --sweep: recompute and check, write nothing")
    parser.add_argument("--processes", type=int, default=12)
    args = parser.parse_args(argv)
    if not args.sweep:
        build()
        return 0
    start = time.perf_counter()
    sweep = run_sweep(args.processes)
    print(f"sweep: {len(sweep['arms']) * 2 * len(SWEEP_SEEDS)} runs in {time.perf_counter() - start:.0f} s")
    if not args.check:
        SWEEP_JSON.write_text(json.dumps(sweep, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {SWEEP_JSON.relative_to(ROOT)}")
        return 0
    recorded = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))
    ok = recorded["train_cell_sha256"] == sweep["train_cell_sha256"]
    print("recorded training-cell hash matches:", ok)
    for name, holds in sweep_claims(sweep).items():
        print(f"{'holds' if holds else 'FAILS'}: {name}")
        ok = ok and holds
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
