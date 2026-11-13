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

# --- Per-chapter constants -----------------------------------------------------------------------------------
# The shared tests read every entry of these through tests/_files.chapter_constants; the chapter changes them
# here and nowhere else.
# SYNC_EXEMPT       name -> reason a definition may differ between the chapter page and its notebook.
#                   Example: {"train_step": "the notebook version also logs the learning rate"}
# RESULT_CLAIMS     (chapter regex, notebook-output regex, comparator) for a number the chapter states and the
#                   notebook prints; comparator eq, gt, lt or approx:<rel>. Captured numbers are normalised, so
#                   10,000 and 1e4 are the same number. Example: (r"loss of (\d+\.\d+)", r"final loss: (\S+)", "approx:0.01")
# NOT_A_RESULT      sentence regex -> reason for a sentence that links a notebook and holds a number but states no
#                   result. Example: {r"parses two files": "a count of inputs, not a result"}
# NOTATION_OK       unit regex -> reason for a sentence the notation check's banned phrases match on purpose.
#                   Example: {r"capacity factor of the paper": "quoted from the Switch paper"}
# MIN_FIGURES       the least number of figures the chapter must have. Example: 12
# INTERFACE_LABELS  label -> text of the heading the label sits directly above, for the section labels other
#                   chapters link to. Example: {"sec-ch2-flops": "Training Complexity"}
# A NOT_A_RESULT or NOTATION_OK entry that matches nothing is a failure: remove it with the sentence it was for.
# The chapter's checker reads every exemption with `python tools/list_exemptions.py chapters/<dir>`.
SYNC_EXEMPT: dict[str, str] = {}  # name -> reason it may differ between the chapter and its notebook

# (chapter regex, notebook-output regex, comparator); each regex has exactly one capture group.
# Comparator: eq, gt, lt or approx:<rel>, applied as `chapter_value <op> notebook_value`.
RESULT_CLAIMS: list[tuple[str, str, str]] = []

# regex over a sentence -> one-line reason; such a sentence links a notebook and mentions a number but states
# no result, so the notebook-link check (rule c of test_notebook_prose.py) does not flag it
NOT_A_RESULT: dict[str, str] = {}

MIN_FIGURES: int = 12
# Units the notation check's banned phrases may match, declared on purpose: unit regex -> reason.
NOTATION_OK: dict[str, str] = {}
# Section labels other chapters link to: label -> the text of the heading it sits directly above.
INTERFACE_LABELS: dict[str, str] = {}


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
        "intermediate_size": 14336, "vocab_size": 256000, "tie_word_embeddings": True},  # vocabulary: hub, not paper
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


def test_gemma2_paper_vocabulary_and_parameter_split():
    """C6-01: the Gemma 2 paper's Table 1 vocabulary (256,128) gives its Table 2 counts; the checkpoint has 256,000."""
    cfg = PAPER_CONFIGS["google/gemma-2-9b"]
    d = cfg["hidden_size"]
    assert 256_128 * d == 917_962_752  # paper, Table 2, embedding parameters (9B)
    assert cfg["vocab_size"] * d == 917_504_000  # released checkpoint
    non_embedding = COUNT(cfg)["total"] - cfg["vocab_size"] * d  # tied embeddings: one table
    assert non_embedding == 8_324_201_984  # paper, Table 2, non-embedding parameters (9B)
    paper_total = COUNT({**cfg, "vocab_size": 256_128})["total"]
    assert paper_total == 9_242_164_736 != HUB_TOTALS["google/gemma-2-9b"]["elements_total"]


def test_yarn_attention_factor_grows_so_temperature_falls():
    """C6-02: sqrt(1/tau) = 0.1 ln s + 1 (YaRN, arXiv v3, Eq. 15) grows with s, so tau falls."""
    import math
    taus = [1 / (0.1 * math.log(s) + 1) ** 2 for s in (1, 2, 8, 40)]
    assert taus[0] == 1 and all(a > b for a, b in zip(taus, taus[1:]))
    assert round(0.1 * math.log(8) + 1, 3) == 1.208  # the paper's worked value for s = 8


SWEEP = DATA / "norm_sweep.json"
NOTEBOOK = DATA.parent / "architecture_demo.ipynb"


def test_norm_sweep_records_the_current_training_cell():
    """D22: the sweep's recorded SHA-256 is that of the notebook's training cell as committed today."""
    import hashlib
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    train_cell = [c for c in cells if c["cell_type"] == "code" and "def train(" in "".join(c["source"])]
    assert len(train_cell) == 1
    digest = hashlib.sha256("".join(train_cell[0]["source"]).encode("utf-8")).hexdigest()
    recorded = json.loads(SWEEP.read_text(encoding="utf-8"))
    assert recorded["train_cell_sha256"] == digest
    assert [(a["lr"], a["warmup"]) for a in recorded["arms"]] == [
        (lr, w) for lr in (3e-3, 1e-3, 3e-4) for w in (0, 100)]
    assert all(len(a["pre"]) == len(a["post"]) == 5 for a in recorded["arms"])


@pytest.mark.live
def test_norm_sweep_check_recomputes_every_arm():
    """D22: recompute all 60 runs and require every statement the notebook asserts from the sweep to hold."""
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "build_architecture_notebook.py"),
                             "--sweep", "--check"], capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
