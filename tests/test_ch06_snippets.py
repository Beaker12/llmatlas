"""Run chapter 6's count_params fence on the committed configurations and check its numbers.

Three independent checks per model: the exact element count of the stored tensors (read from the
safetensors headers by tools/fetch_arch_samples.py), the vendor's published figures under the
vendor's own convention for "active" parameters, and exact regression pins.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "06-architectures" / "index.md"
DATA = ROOT / "chapters" / "06-architectures" / "examples" / "data"
FENCE = re.compile(r"```python\n([\s\S]*?)```")


def chapter_function(marker: str) -> dict:
    for m in FENCE.finditer(CHAPTER.read_text(encoding="utf-8")):
        if marker in m.group(1):
            namespace: dict = {}
            exec(m.group(1), namespace)  # noqa: S102 - executing the book's own example
            return namespace
    raise AssertionError(f"no python fence in chapter 6 contains {marker!r}")


COUNT = chapter_function("def count_params")["count_params"]
CHECKPOINT = json.loads((DATA / "checkpoint_counts.json").read_text())
HUB_TOTALS = json.loads((DATA / "hub_totals.json").read_text())


def config(repo: str) -> dict:
    return json.loads((DATA / "configs" / (repo.replace("/", "__") + ".json")).read_text())


# repo: (total in billions, tolerance, active in billions, convention, tolerance)
PUBLISHED = {
    "Qwen/Qwen2.5-7B": (7.61, 0.01, None, None, None),  # vendor truncates 7.6156
    "allenai/OLMo-2-1124-7B": (7.3, 0.05, None, None, None),
    "allenai/OLMoE-1B-7B-0924": (6.9, 0.05, 1.3, "with", 0.05),
    "deepseek-ai/DeepSeek-V2-Lite": (15.7, 0.05, 2.4, "without", 0.1),  # vendor truncates 2.451
    "deepseek-ai/DeepSeek-V3": (671, 0.5, 37, "without", 0.5),
    "mistralai/Mistral-7B-v0.1": (7.3, 0.1, None, None, None),  # vendor rounds 7.2417 up
    "mistralai/Mixtral-8x7B-v0.1": (46.7, 0.05, 12.9, "with", 0.05),
}

# Exact results, so that any edit to the function that changes a count fails here
PINS = {
    "Qwen/Qwen2.5-7B": (7615616512, 7615616512, 7070619136),
    "allenai/OLMo-2-1124-7B": (7298617344, 7298617344, 6887575552),
    "allenai/OLMoE-1B-7B-0924": (6919161856, 1282017280, 1178994688),
    "deepseek-ai/DeepSeek-V2-Lite": (15706484224, 2661150208, 2451435008),
    "deepseek-ai/DeepSeek-V3": (671026419200, 37552297472, 36625618432),
    "mistralai/Mistral-7B-v0.1": (7241732096, 7241732096, 7110660096),
    "mistralai/Mixtral-8x7B-v0.1": (46702792704, 12879925248, 12748853248),
}

# Architectures transcribed from the papers of repositories that are gated
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
        "intermediate_size": 14336, "vocab_size": 256000, "tie_word_embeddings": True},
}


@pytest.mark.parametrize("repo", sorted(PINS))
def test_total_matches_the_stored_tensors_exactly(repo):
    assert COUNT(config(repo))["total"] == CHECKPOINT[repo]["elements_counted"]


@pytest.mark.parametrize("repo", sorted(PAPER_CONFIGS))
def test_paper_configs_match_the_hub_totals_exactly(repo):
    assert COUNT(PAPER_CONFIGS[repo])["total"] == HUB_TOTALS[repo]["elements_total"]


@pytest.mark.parametrize("repo", sorted(PUBLISHED))
def test_published_figures_under_each_vendors_convention(repo):
    total, tol, active, convention, tol_active = PUBLISHED[repo]
    counts = COUNT(config(repo))
    assert abs(counts["total"] / 1e9 - total) <= tol
    if active is not None:
        other = "without" if convention == "with" else "with"
        assert abs(counts[f"active_{convention}_embedding"] / 1e9 - active) <= tol_active
        assert abs(counts[f"active_{other}_embedding"] / 1e9 - active) > tol_active


@pytest.mark.parametrize("repo", sorted(PINS))
def test_regression_pins(repo):
    counts = COUNT(config(repo))
    got = (counts["total"], counts["active_with_embedding"], counts["active_without_embedding"])
    assert got == PINS[repo]


def test_coverage_spans_dense_sparse_and_latent_attention():
    kinds = {config(repo)["model_type"] for repo in PINS}
    assert {"qwen2", "mixtral", "deepseek_v3", "deepseek_v2"} <= kinds


def test_unsupported_model_type_raises():
    with pytest.raises(ValueError, match="unsupported model_type"):
        COUNT({"model_type": "gpt2"})


def test_kv_cache_function_matches_the_chapter_table():
    kv = chapter_function("def kv_cache_elements_per_token")["kv_cache_elements_per_token"]
    assert 2 * kv(config("deepseek-ai/DeepSeek-V3")) == 70272
    assert 2 * kv(PAPER_CONFIGS["meta-llama/Llama-3.1-8B"]) == 131072
    assert 2 * kv(config("allenai/OLMo-2-1124-7B")) == 524288
