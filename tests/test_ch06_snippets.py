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
RESULT_CLAIMS: list[tuple[str, str, str]] = [
    (r"element counts of all (\d+) downloadable checkpoints", r"reproduced exactly for (\d+) models", "eq"),
    (r"multi-head cache is (\d+) times Llama 3\.1 8B's", r"multi-head cache is (\d+) times Llama 3\.1 8B's", "eq"),
    (r"auxiliary loss is near (\d+) for a balanced router", r"load-balancing loss (\S+) balanced", "approx:0.1"),
]

# regex over a sentence -> one-line reason; such a sentence links a notebook and mentions a number but states
# no result, so the notebook-link check (rule c of test_notebook_prose.py) does not flag it
NOT_A_RESULT: dict[str, str] = {
    r"post-norm fails without warm-up only at the highest of three rates":
        "'three' counts the sweep's learning rates, a setting; the result is qualitative and asserted by "
        "norm_claims in the sweep and summary cells",
    r"trains both arrangements side by side at three learning rates":
        "describes the experiment's settings (three learning rates), states no result",
    r"trains a six-layer post-norm and pre-norm model\s+from\s+the same seeds at three learning rates":
        "the numbers are settings (depth, number of rates, 'two lower rates'); the outcome is qualitative and "
        "asserted by norm_claims",
    r"asserts the identities above to \$10\^\{-9\}\$":
        "the only numbers are the assertion's tolerance and symbol subscripts; the identities are asserted in "
        "the rope cell",
    r"The notebook computes the ratio for all twelve models":
        "a pointer; 'twelve' counts the models, and the ratios are printed and asserted in the notebook cell",
    r"scaling \$W_Q\$ and \$W_K\$ by 100 multiplies":
        "the scale 100 is the check's setting; the factor 10^4 and the invariance are asserted in the QK-norm cell",
    r"KV cache per token, all layers, bf16, in KiB":
        "caption's unit definition (1 KiB = 1,024 bytes) and model facts (61 layers, 128 heads, 4 key/value "
        "heads) from the configurations; its one result, the factor 4, is a RESULT_CLAIMS entry",
}

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
    "Qwen/Qwen2.5-7B": (7.61, 0.01, None, None, None),  # consistent with truncating 7.6156
    "Qwen/Qwen3-8B-Base": (8.2, 0.05, None, None, None),  # model card at the pinned commit
    "allenai/OLMoE-1B-7B-0924": (6.9, 0.05, 1.3, "with", 0.05),  # paper: "6.9B total parameters"
    "deepseek-ai/DeepSeek-V2-Lite": (15.7, 0.05, 2.4, "without", 0.1),  # vendor truncates 2.451
    "deepseek-ai/DeepSeek-V3": (671, 0.5, 37, "without", 0.5),
    "mistralai/Mistral-7B-v0.1": (7.3, 0.1, None, None, None),  # consistent with rounding 7.2417 up
    "mistralai/Mixtral-8x7B-v0.1": (46.7, 0.05, 12.9, "with", 0.05),
}

# Exact results, so that any edit to the function that changes a count fails here
PINS = {
    "Qwen/Qwen2.5-7B": (7615616512, 7615616512, 7070619136),
    "Qwen/Qwen3-8B-Base": (8190735360, 8190735360, 7568405504),
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

# Gemma 3 12B, text decoder only: shapes from the Gemma3_12B configuration of Google's reference code
# (gemma/gm/nn/_gemma.py, which has 262,144 embedding rows); vocabulary 262,208, the released checkpoint's rows
# (Transformers' convert_gemma3_weights.py appends 64); the hub total includes vision.
GEMMA3 = {
    "model_type": "gemma3_text", "hidden_size": 3840, "num_hidden_layers": 48,
    "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
    "intermediate_size": 15360, "vocab_size": 262208, "tie_word_embeddings": True}

# Gemma 3 12B's hub total in three parts: the text decoder with the checkpoint's 262,208 embedding rows, a SigLIP
# encoder at 896 pixels without its pooling head, and the projection into the text width with its norm.
SIGLIP_WIDTH, SIGLIP_FFN, SIGLIP_LAYERS, PATCH = 1152, 4304, 27, 14
SIGLIP_LAYER = (4 * (SIGLIP_WIDTH ** 2 + SIGLIP_WIDTH) + 2 * SIGLIP_WIDTH * SIGLIP_FFN + SIGLIP_FFN + SIGLIP_WIDTH
                + 2 * 2 * SIGLIP_WIDTH)  # q, k, v, o with biases; two-matrix MLP with biases; two LayerNorms
SIGLIP = (SIGLIP_LAYERS * SIGLIP_LAYER + 3 * PATCH ** 2 * SIGLIP_WIDTH + SIGLIP_WIDTH  # patch embedding
          + (896 // PATCH) ** 2 * SIGLIP_WIDTH + 2 * SIGLIP_WIDTH)  # 4,096 positions, final LayerNorm
PROJECTION = SIGLIP_WIDTH * 3840 + SIGLIP_WIDTH


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
    assert {"qwen2", "qwen3", "mixtral", "deepseek_v3", "deepseek_v2"} <= kinds


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


def test_gemma3_text_decoder_matches_the_papers_table_1():
    """D23: the paper's Table 1 gives 10,759M non-embedding and a 417M vision encoder for the 12B model."""
    counts = COUNT(GEMMA3)
    assert counts["total"] == 11_766_034_176
    non_embedding = counts["total"] - GEMMA3["vocab_size"] * GEMMA3["hidden_size"]
    assert non_embedding == 10_759_155_456 and round(non_embedding / 1e6) == 10_759
    assert SIGLIP == 416_866_032 and round(SIGLIP / 1e6) == 417 and PROJECTION == 4_424_832
    hub = HUB_TOTALS["google/gemma-3-12b-pt"]["elements_total"]
    assert counts["total"] + SIGLIP + PROJECTION == hub == 12_187_325_040  # exact, three parts
    assert non_embedding + 262_144 * 3840 + SIGLIP + PROJECTION == hub - 245_760  # reference code's rows fall short
    for rows in (262_144, 262_208):  # the paper's 1,012M embedding is reproduced by neither row count
        assert round(rows * 3840 / 1e6) != 1012


def test_qwen3_non_embedding_and_kv_cache():
    """D23: Qwen3 8B's model card gives 6.95B non-embedding; its cache is 73,728 elements per token."""
    cfg = config("Qwen/Qwen3-8B-Base")
    non_embedding = COUNT(cfg)["total"] - 2 * cfg["vocab_size"] * cfg["hidden_size"]
    assert non_embedding == 6_946_075_648 and round(non_embedding / 1e9, 2) == 6.95
    kv = chapter_function("def kv_cache_elements_per_token")["kv_cache_elements_per_token"]
    assert kv(cfg) == 73_728 and kv(GEMMA3) == 196_608


def test_gemma3_cache_at_128k_with_five_local_layers_per_global():
    """D23: 8 global and 40 local (1,024-token) layers; bf16 bytes at 131,072 tokens."""
    per_layer = 2 * 8 * 256
    global_bytes, local_bytes = 8 * per_layer * 131_072 * 2, 40 * per_layer * 1_024 * 2
    assert (round(global_bytes / 1e9, 2), round(local_bytes / 1e9, 2)) == (8.59, 0.34)
    assert round((global_bytes + local_bytes) / 1e9, 2) == 8.93
    assert round(48 * per_layer * 131_072 * 2 / 1e9, 1) == 51.5


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


def test_norm_sweep_caption_counts():
    """The fig-ch6-norm-sweep caption: without warm-up post-norm is below pre-norm in every seed at the two lower
    rates and fails (stays above 3.0 nats) in every seed at 3e-3, where pre-norm trains (below 2.5); with the
    100-step warm-up, post-norm at 3e-3 trains in four seeds and fails in one."""
    arms = {(a["lr"], a["warmup"]): a for a in json.loads(SWEEP.read_text(encoding="utf-8"))["arms"]}
    for lr in (1e-3, 3e-4):
        assert all(post < pre for pre, post in zip(arms[(lr, 0)]["pre"], arms[(lr, 0)]["post"]))
    assert all(v > 3.0 for v in arms[(3e-3, 0)]["post"]) and all(v < 2.5 for v in arms[(3e-3, 0)]["pre"])
    warm = arms[(3e-3, 100)]["post"]
    assert (sum(v < 2.5 for v in warm), sum(v > 3.0 for v in warm)) == (4, 1)


@pytest.mark.live
def test_norm_sweep_check_recomputes_every_arm():
    """D22: recompute all 60 runs and require every statement the notebook asserts from the sweep to hold."""
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "build_architecture_notebook.py"),
                             "--sweep", "--check"], capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def test_notebook_plots_are_embedded():
    """Every plotting cell of the committed notebook, and every cell tagged figure:<slug>, carries an image/png
    output: the notebook must be executed for commit with MPLBACKEND unset, or the published page shows no plots."""
    cells = json.loads((ROOT / "chapters" / "06-architectures" / "examples" / "architecture_demo.ipynb")
                       .read_text(encoding="utf-8"))["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]

    def has_image(cell):
        return any("image/png" in out.get("data", {}) for out in cell.get("outputs", []))

    plotting = [c for c in code if "plt.show()" in "".join(c["source"])]
    tagged = [c for c in code if any(t.startswith("figure:") for t in c["metadata"].get("tags", []))]
    assert len(plotting) >= 5 and tagged
    assert all(has_image(c) for c in plotting + tagged)
    assert sum(has_image(c) for c in code) >= len(plotting)


@pytest.mark.slow
def test_make_figures_is_deterministic(tmp_path):
    """Two runs of figures/make_figures.py write byte-identical files (fixed salt, no dates, glyphs as paths)."""
    import os
    import subprocess
    import sys
    script = ROOT / "chapters" / "06-architectures" / "figures" / "make_figures.py"
    runs = []
    for name in ("first", "second"):
        out = tmp_path / name
        done = subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True,
                              env={**os.environ, "MPLBACKEND": "Agg"}, timeout=900)
        assert done.returncode == 0, done.stderr[-2000:]
        runs.append({p.name: p.read_bytes() for p in sorted(out.glob("*.svg"))})
    assert len(runs[0]) == 7 and runs[0] == runs[1]


# --- The router, frequency and grouped-query fences, executed ------------------------------------------------
def test_grouped_query_fence_equals_attention_with_repeated_heads():
    """The 4.3 fence: GQA equals ordinary attention per query head over its group's key/value head; n_kv = 1 is
    MQA."""
    import torch
    import torch.nn.functional as F
    gqa = chapter_function("def grouped_query_attention")["grouped_query_attention"]
    torch.manual_seed(0)
    q = torch.randn(2, 8, 5, 16, dtype=torch.float64)
    for n_kv in (8, 2, 1):
        k, v = torch.randn(2, n_kv, 5, 16, dtype=torch.float64), torch.randn(2, n_kv, 5, 16, dtype=torch.float64)
        group = 8 // n_kv
        expected = torch.cat([F.scaled_dot_product_attention(q[:, h:h + 1], k[:, h // group:h // group + 1],
                                                             v[:, h // group:h // group + 1], is_causal=True)
                              for h in range(8)], dim=1)
        assert torch.allclose(gqa(q, k, v), expected, atol=1e-12)


def test_router_fence_gates_and_auxiliary_loss():
    """The 6.1 fence: gates renormalize over the chosen experts; the loss equals E * sum f_i P_i computed by
    hand, is 1 for a uniform router and well above 1 when every token prefers one expert."""
    import torch
    route = chapter_function("def route_top_k")["route_top_k"]
    torch.manual_seed(0)
    x = torch.randn(512, 16, dtype=torch.float64)
    weight = torch.randn(8, 16, dtype=torch.float64)
    ids, gates, loss = route(x, weight, 2)
    assert ids.shape == gates.shape == (512, 2)
    assert torch.allclose(gates.sum(-1), torch.ones(512, dtype=torch.float64))
    probs = (x @ weight.T).softmax(-1)
    f = torch.bincount(ids.flatten(), minlength=8).double() / ids.numel()
    assert torch.isclose(loss, 8 * (f * probs.mean(0)).sum())
    assert route(torch.ones(64, 16), torch.zeros(8, 16), 2)[2].item() == 1.0
    favored = torch.zeros(8, 16)
    favored[0] = 1.0  # every token prefers expert 0
    assert route(torch.ones(64, 16), favored, 2)[2].item() > 1.5


def test_frequency_fences():
    """The 5.2 fences: YaRN keeps 21, blends 25 and interpolates 18 pairs at d_k = 128, s = 8, and 11/12/9 with
    DeepSeek-V3's committed rope settings; the NTK-aware base keeps omega_0 and divides the lowest by s."""
    import math
    yarn = chapter_function("def yarn_frequencies")["yarn_frequencies"]
    rope = chapter_function("def rope_frequencies")
    v3 = config("deepseek-ai/DeepSeek-V3")
    sc = v3["rope_scaling"]
    cases = (((128, 10000.0, 4096, 8), (21, 25, 18)),
             ((v3["qk_rope_head_dim"], float(v3["rope_theta"]), sc["original_max_position_embeddings"],
               sc["factor"], sc["beta_slow"], sc["beta_fast"]), (11, 12, 9)))
    for args, expected in cases:
        pairs = yarn(*args)
        kept, interp = sum(p == 1.0 for _, p in pairs), sum(p == 0.0 for _, p in pairs)
        assert (kept, len(pairs) - kept - interp, interp) == expected
        assert math.isclose(args[1] ** (-(args[0] - 2) / args[0]) / pairs[-1][0], args[3])
    for s in (2, 8, 40):
        old = rope["rope_frequencies"](128)
        new = rope["rope_frequencies"](128, rope["ntk_aware_base"](10000.0, s, 128))
        assert math.isclose(new[0].item(), old[0].item()) and math.isclose(old[-1].item() / new[-1].item(), s)


# --- Exercise answers ----------------------------------------------------------------------------------------
# Each solution's listing is run from the page; ANSWERS are patterns it must print, and STATED are numbers the
# prose of the answer gives, which must appear in the printed output too. The inputs the solutions transcribe
# are checked against the committed configurations and the counting function in the last test.
SOLUTION = re.compile(r"^`{4}\{admonition\} Solution (\d+)\n(.*?)^`{4}$", re.S | re.M)
LISTING = re.compile(r"^```python\n(.*?)^```$", re.S | re.M)
ANSWERS = {
    1: [r"n_kv = 32: 524,288 bytes per token, 17\.18 GB per sequence,  0 fit",
        r"n_kv = 16: 262,144 bytes .* 1 fit", r"n_kv =  8: 131,072 bytes .* 3 fit", r"n_kv =  1: .* 29 fit"],
    2: [r"removed: 9,395,240,960", r"2,621,440 bytes before, 327,680 after", r"scores unchanged: True",
        r"relative change of the scores: 0\.93"],
    3: [r"biases: 129,024; count without them: 7,615,487,488", r"7\.61B with, 7\.61B without"],
    4: [r"agree: True", r"without absorption 268,959,744, with 2,162,688: 124 times fewer"],
    5: [r"kept 11, blended 12, interpolated 9", r"divided by 40\n", r"0\.5462 rad in training, 0\.5462 rad at 163,840"],
    6: [r"phi = 1\.0: capacity 1,024, expert 1 drops 1,024", r"phi = 1\.25: capacity 1,280, expert 1 drops 768",
        r"skewed router: 1\.143", r"uniform router: 1\n"],
    7: [r"most loaded 1\.97, least loaded 0\.348", r"200 batches: most loaded 1\.16, least loaded 0\.857"],
    8: [r"Mixtral 8x7B +with embedding 27\.58%, without 27\.30%", r"OLMoE 1B-7B +with embedding 18\.53%, without 17\.04%",
        r"V2-Lite with embedding 16\.94%, without 15\.61%", r"DeepSeek-V3 +with embedding +5\.60%, without +5\.46%"],
}
STATED = {1: ["17.18", "29"], 2: ["2,621,440", "327,680"], 3: ["129,024", "7.61"],
          4: ["268,959,744", "2,162,688", "124"], 5: ["0.5462", "163,840"], 6: ["1,024", "1,280", "768", "1.143"],
          7: ["1.97", "0.348", "1.16", "0.857"], 8: ["5.60%", "5.46%", "18.53%", "17.04%"]}


def solutions() -> dict:
    return {int(n): body for n, body in SOLUTION.findall(CHAPTER.read_text(encoding="utf-8"))}


def test_every_exercise_has_its_answers_listed():
    assert sorted(solutions()) == sorted(ANSWERS) == list(range(1, 9))


@pytest.mark.parametrize("n", sorted(ANSWERS))
def test_exercise_answer(n):
    import os
    import subprocess
    import sys
    body = solutions()[n]
    listings = LISTING.findall(body)
    assert len(listings) == 1, f"solution {n} should hold one listing"
    run = subprocess.run([sys.executable, "-W", "error", "-c", listings[0]], capture_output=True, text=True,
                         timeout=600, env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stderr[-2000:]
    for pattern in ANSWERS[n]:
        assert re.search(pattern, run.stdout), f"solution {n}: {pattern!r} not in\n{run.stdout}"
    answer = LISTING.sub("", body).split("% output")[-1]
    for number in STATED[n]:
        assert number in answer, f"solution {n}'s answer no longer states {number}"
        assert number in run.stdout, f"solution {n} states {number} but the listing prints\n{run.stdout}"


def test_exercise_inputs_match_the_configurations_and_the_counter():
    """What the solutions transcribe: Qwen2.5's shape and checkpoint count (solution 3), and solution 8's
    totals, active counts and input-embedding sizes, recomputed by count_params from the committed configs."""
    qwen = config("Qwen/Qwen2.5-7B")
    head = qwen["hidden_size"] // qwen["num_attention_heads"]
    assert (qwen["num_hidden_layers"], qwen["hidden_size"], qwen["num_key_value_heads"], head) == (28, 3584, 4, 128)
    assert CHECKPOINT["Qwen/Qwen2.5-7B"]["elements_counted"] == COUNT(qwen)["total"] == 7_615_616_512
    for repo, total, active, key, vocab, width in (
            ("mistralai/Mixtral-8x7B-v0.1", 46_702_792_704, 12_879_925_248, "active_with_embedding", 32_000, 4_096),
            ("allenai/OLMoE-1B-7B-0924", 6_919_161_856, 1_282_017_280, "active_with_embedding", 50_304, 2_048),
            ("deepseek-ai/DeepSeek-V2-Lite", 15_706_484_224, 2_451_435_008, "active_without_embedding", 102_400,
             2_048),
            ("deepseek-ai/DeepSeek-V3", 671_026_419_200, 36_625_618_432, "active_without_embedding", 129_280,
             7_168)):
        cfg = config(repo)
        counts = COUNT(cfg)
        assert (cfg["vocab_size"], cfg["hidden_size"]) == (vocab, width)
        assert (counts["total"], counts[key]) == (total, active)
        assert counts["active_with_embedding"] - counts["active_without_embedding"] == vocab * width


def _yarn_reference_counts(dim, base, n_train, beta_fast=32.0, beta_slow=1.0):
    """Kept, blended and interpolated pairs under the ramp in pair index of DeepSeek-V3's inference code
    (precompute_freqs_cis: floor/ceil of the correction dimensions, then a linear ramp in the pair index)."""
    import math
    def correction(turns):
        return dim * math.log(n_train / (turns * 2 * math.pi)) / (2 * math.log(base))
    low, high = max(math.floor(correction(beta_fast)), 0), min(math.ceil(correction(beta_slow)), dim - 1)
    ramp = [min(max((i - low) / (high - low), 0.0), 1.0) for i in range(dim // 2)]
    return (sum(r == 0 for r in ramp), sum(0 < r < 1 for r in ramp), sum(r == 1 for r in ramp)), (low, high)


def test_yarn_reference_ramp_in_pair_index_agrees_on_the_counts():
    """5.2: the index-form ramp keeps pairs 0-20 and interpolates 46-63 (21, 25, 18), as the page's xi form does;
    Exercise 1's DeepSeek-V3 rotary dimensions give 11, 12, 9 both ways."""
    assert _yarn_reference_counts(128, 10000.0, 4096) == ((21, 25, 18), (20, 46))
    assert _yarn_reference_counts(64, 10000.0, 4096)[0] == (11, 12, 9)
    text = CHAPTER.read_text(encoding="utf-8")
    assert "keeps pairs 0 to 20 and fully interpolates pairs 46 to 63, again 21, 25 and 18" in text


def test_solution2_multi_head_base():
    """Solution 2: the multi-head model of Llama 3.1 70B's shape has 79.9B parameters; GQA removes 9.4B of them."""
    gqa = 70_553_706_496
    extra = 80 * 2 * 8192 * (64 - 8) * 128
    assert round((gqa + extra) / 1e9, 2) == 79.95 and round(extra / 1e9, 2) == 9.40 and round(gqa / 1e9, 2) == 70.55
    assert "removes 9.40 billion of the 79.95 billion parameters" in CHAPTER.read_text(encoding="utf-8")
