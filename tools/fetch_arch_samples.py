#!/usr/bin/env python3
"""Fetch the committed samples for the chapter 6 notebook (standard library only).

For each candidate model repository on the Hugging Face hub this script

1. pins the current commit (``data/pins.json``; an existing file is reused so that re-runs
   fetch the same bytes),
2. saves ``config.json`` at that commit to ``data/configs/<org>__<name>.json``,
3. reads the header of every ``.safetensors`` shard with two HTTP range requests (8 bytes
   for the header length, then the JSON header) and sums tensor element counts, so that the
   count does not depend on any byte width (OLMo 2 ships float32, DeepSeek-V3 mixes BF16,
   FP8 and F32), and writes ``data/checkpoint_counts.json``,
4. saves a public-domain training text, and
5. writes ``data/checksums.json`` with the sha256 of every file in ``data/``.

Usage: python tools/fetch_arch_samples.py chapters/06-architectures/examples/data
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import pathlib
import re
import struct
import sys
import time
import urllib.request

UA = {"User-Agent": "llmatlas-notebook/0.1"}
HUB = "https://huggingface.co"

CANDIDATES = [
    "Qwen/Qwen2.5-7B",
    "allenai/OLMo-2-1124-7B",
    "allenai/OLMoE-1B-7B-0924",
    "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-ai/DeepSeek-V3",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
]

# Gated repositories: the config is not downloadable without signing in, but the hub's public
# metadata still reports the exact parameter total of the checkpoint.
GATED = ["meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-70B", "google/gemma-2-9b"]

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
TEXT_BYTES = 150_000


def get(url, headers=None, timeout=30, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, **(headers or {})})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def slug(repo):
    return repo.replace("/", "__")


def pin_commits(out):
    path = out / "pins.json"
    if path.exists():
        return json.loads(path.read_text())
    pins = {}
    for repo in CANDIDATES:
        info = json.loads(get(f"{HUB}/api/models/{repo}"))
        pins[repo] = info["sha"]
    path.write_text(json.dumps(pins, indent=1) + "\n")
    return pins


def shard_names(repo, sha):
    """Shard file names: from the index file when sharded, else the single checkpoint."""
    try:
        index = json.loads(get(f"{HUB}/{repo}/resolve/{sha}/model.safetensors.index.json"))
        return sorted(set(index["weight_map"].values()))
    except Exception:  # noqa: BLE001
        return ["model.safetensors"]


def read_header(repo, sha, shard):
    base = f"{HUB}/{repo}/resolve/{sha}/{shard}"
    (length,) = struct.unpack("<Q", get(base, {"Range": "bytes=0-7"}))
    return json.loads(get(base, {"Range": f"bytes=8-{7 + length}"}))


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def count_repo(repo, sha, num_layers):
    names = shard_names(repo, sha)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        headers = list(pool.map(lambda s: read_header(repo, sha, s), names))
    total, by_dtype, mtp, scales = 0, {}, 0, 0
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    for header in headers:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            n = numel(meta["shape"])
            by_dtype[meta["dtype"]] = by_dtype.get(meta["dtype"], 0) + n
            m = layer_re.search(name)
            if name.endswith("_scale_inv"):
                scales += n
            elif m and int(m.group(1)) >= num_layers:
                mtp += n
            else:
                total += n
    return {
        "sha": sha,
        "shards": len(names),
        "elements_counted": total,
        "elements_mtp_layer": mtp,
        "elements_fp8_scales": scales,
        "elements_by_dtype": by_dtype,
    }


def fetch_training_text():
    raw = get(GUTENBERG_URL, timeout=60).decode("utf-8-sig").replace("\r\n", "\n")
    start = raw.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    body = raw[start:]
    body = body[body.index("It is a truth universally acknowledged") :]
    body = re.sub(r"\[Illustration[^\]]*\]", "", body)
    for old, new in {"‘": "'", "’": "'", "“": '"', "”": '"',
                     "–": "-", "—": "--", " ": " ", "ê": "e", "à": "a",
                     "é": "e", "œ": "oe"}.items():
        body = body.replace(old, new)
    body = re.sub(r"\n{3,}", "\n\n", body)
    text = body[:TEXT_BYTES].rsplit("\n", 1)[0] + "\n"
    assert text.isascii(), "training text must be ASCII"
    return text.encode("ascii")


def main(out):
    out = pathlib.Path(out)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    pins = pin_commits(out)
    counts_path = out / "checkpoint_counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    for repo in CANDIDATES:
        sha = pins[repo]
        raw = get(f"{HUB}/{repo}/resolve/{sha}/config.json")
        (out / "configs" / f"{slug(repo)}.json").write_bytes(raw)
        cfg = json.loads(raw)
        if repo not in counts or counts[repo]["sha"] != sha:  # reuse counts for the same commit
            counts[repo] = count_repo(repo, sha, cfg["num_hidden_layers"])
        c = counts[repo]
        print(f"{repo:32} {c['shards']:4} shards  counted {c['elements_counted']:>15,}  "
              f"mtp {c['elements_mtp_layer']:>13,}  fp8 scales {c['elements_fp8_scales']:>12,}")
    (out / "checkpoint_counts.json").write_text(json.dumps(counts, indent=1) + "\n")

    totals = {}
    for repo in GATED:
        info = json.loads(get(f"{HUB}/api/models/{repo}"))
        st = info["safetensors"]
        totals[repo] = {"sha": info["sha"], "elements_total": st["total"], "by_dtype": st["parameters"]}
        print(f"{repo:32} gated; hub metadata total {st['total']:,}")
    (out / "hub_totals.json").write_text(json.dumps(totals, indent=1) + "\n")

    (out / "pride_and_prejudice_150k.txt").write_bytes(fetch_training_text())

    sums = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (out / "checksums.json").write_text(
        json.dumps({"generated": time.strftime("%Y-%m-%d"), "sha256": sums}, indent=1) + "\n"
    )
    print(f"{len(sums)} files checksummed; data directory "
          f"{sum(p.stat().st_size for p in out.rglob('*') if p.is_file()):,} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
