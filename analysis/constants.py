from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
ORIGINAL_PAPER_DIR = DATA_DIR / "original_paper"
ORIGINAL_PAPER_FILE = ORIGINAL_PAPER_DIR / "DataExpanded_FINAL.csv.gz"
FIGURES_DIR = ROOT / "nature_medicine_paper" / "figures"
SUMMARY_PATH = ROOT / "analysis_summary.json"

BOOTSTRAP_DRAWS = 500
BOOTSTRAP_SEED = 0
CALIBRATION_BINS = 10

UTILITY_COSTS: list[tuple[int, int]] = [
    (100, 1),
    (10, 1),
    (5, 1),
    (1, 1),
    (1, 5),
    (1, 10),
    (1, 100),
]
FIXED_UTILITY_RATIOS = (0.2, 1.0, 5.0)


def utility_regime(c_fp: int, c_fn: int) -> str:
    return f"decision_u_fp{c_fp}_fn{c_fn}"


def threshold_regime(c_fp: int, c_fn: int) -> str:
    return f"decision_t_fp{c_fp}_fn{c_fn}"


def ratio_from_cost(c_fp: int, c_fn: int) -> float:
    return c_fn / c_fp


def threshold_from_cost(c_fp: int, c_fn: int) -> float:
    return c_fp / (c_fp + c_fn)


UTILITY_REGIMES = [utility_regime(fp, fn) for fp, fn in UTILITY_COSTS]
THRESHOLD_REGIMES = [threshold_regime(fp, fn) for fp, fn in UTILITY_COSTS]
TARGET_RATIOS = {utility_regime(fp, fn): ratio_from_cost(fp, fn) for fp, fn in UTILITY_COSTS}
PROMPTED_THRESHOLDS = {
    threshold_regime(fp, fn): threshold_from_cost(fp, fn) for fp, fn in UTILITY_COSTS
}
DECISION_REGIMES = ["decision_baseline", *UTILITY_REGIMES, *THRESHOLD_REGIMES]

FAMILY_ORDER = [
    "gpt-5-mini",
    "gpt-5.4",
    "DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro",
    "Claude-Fable-5",
    "Claude-Sonnet-5",
]
FAMILY_LABEL = {
    "gpt-5-mini": "GPT-5-mini",
    "gpt-5.4-mini": "GPT-5.4-mini",
    "gpt-5.4": "GPT-5.4",
    "DeepSeek-V4-Flash": "DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
    "Claude-Fable-5": "Claude Fable 5",
    "Claude-Sonnet-5": "Claude Sonnet 5",
}
FAMILY_CMAP = {
    "gpt-5-mini": "Blues",
    "gpt-5.4-mini": "Greens",
    "gpt-5.4": "Oranges",
    "DeepSeek-V4-Flash": "Purples",
    "DeepSeek-V4-Pro": "Greens",
    "Claude-Fable-5": "Reds",
    "Claude-Sonnet-5": "Greys",
}
FAMILY_MARKER = {
    "gpt-5-mini": "o",
    "gpt-5.4-mini": "s",
    "gpt-5.4": "^",
    "DeepSeek-V4-Flash": "D",
    "DeepSeek-V4-Pro": "P",
    "Claude-Fable-5": "X",
    "Claude-Sonnet-5": "v",
}
FAMILY_REASONS = {
    "gpt-5-mini": ["minimal", "medium", "high"],
    "gpt-5.4-mini": ["none", "medium", "high"],
    "gpt-5.4": ["none", "medium", "high"],
    "DeepSeek-V4-Flash": ["none", "high", "max"],
    "DeepSeek-V4-Pro": ["none", "high", "max"],
    "Claude-Fable-5": ["low", "medium", "high"],
    "Claude-Sonnet-5": ["low", "medium", "high"],
}
REASON_LABEL = {
    "minimal": "Minimal",
    "none": "None",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "max": "Max",
}
TIER_LABELS = ("Low/None", "Medium", "High")
SHADE_BY_RANK = (0.55, 0.75, 0.98)
LINESTYLE_BY_RANK = (":", "--", "-")

PAPER_CONFIG_ORDER = [
    "gpt-5-mini_re-minimal",
    "gpt-5-mini_re-medium",
    "gpt-5-mini_re-high",
    "gpt-5.4_re-none",
    "gpt-5.4_re-medium",
    "gpt-5.4_re-high",
    "DeepSeek-V4-Flash_re-none",
    "DeepSeek-V4-Flash_re-high",
    "DeepSeek-V4-Flash_re-max",
    "DeepSeek-V4-Pro_re-none",
    "DeepSeek-V4-Pro_re-high",
    "DeepSeek-V4-Pro_re-max",
    "Claude-Fable-5_re-low",
    "Claude-Fable-5_re-medium",
    "Claude-Fable-5_re-high",
    "Claude-Sonnet-5_re-low",
    "Claude-Sonnet-5_re-medium",
    "Claude-Sonnet-5_re-high",
]

PROVIDER_GROUPS = (
    ("gpt", "GPT models", ("gpt-5-mini", "gpt-5.4")),
    ("deepseek", "DeepSeek models", ("DeepSeek-V4-Flash", "DeepSeek-V4-Pro")),
    ("claude", "Claude models", ("Claude-Fable-5", "Claude-Sonnet-5")),
)

FIG1_CONFIGS = (
    "gpt-5-mini_re-high",
    "gpt-5.4_re-high",
    "DeepSeek-V4-Pro_re-high",
    "Claude-Fable-5_re-high",
)

EXPECTED_FIGURES = [
    "fig1_emergency_triage_performance.png",
    "fig1_original_emergency_triage_performance.png",
    "fig1b_default_utilities.png",
    "fig2_recovered_capability.png",
    "fig3_roc_grid_gpt.png",
    "fig3_roc_grid_deepseek.png",
    "fig3_roc_grid_claude.png",
    "fig3_original_roc_grid_gpt.png",
    "fig3_original_roc_grid_deepseek.png",
    "fig3_original_roc_grid_claude.png",
    "decision_belief_consistency.png",
    "nature_decision_consistency.png",
    "fig4_recovered_threshold.png",
    "fig_belief_distributions.png",
    "fig_original_belief_distributions.png",
    "fig_calibration_curves.png",
    "fig_original_calibration_curves.png",
    "fig_prompt_context.png",
    "fig_prompt_belief.png",
    "fig_prompt_baseline.png",
    "fig_prompt_utility.png",
    "fig_prompt_threshold.png",
]

RATIO_COLORS = {
    0.01: "#08306B",
    0.1: "#08519C",
    0.2: "#3182BD",
    1.0: "#7F7F7F",
    5.0: "#FB6A4A",
    10.0: "#CB181D",
    100.0: "#67000D",
}
FIXED_RATIO_COLORS = {
    0.2: "#31A354",
    1.0: "#E7298A",
    5.0: "gold",
}


def parse_config(name: str) -> tuple[str, str]:
    return name.rsplit("_re-", 1)


def reason_rank(family: str, reason: str) -> int:
    return FAMILY_REASONS[family].index(reason)


def ratio_key(value: float) -> str:
    return f"{value:g}"


def ratio_label(value: float) -> str:
    if value >= 1:
        return f"{value:g}"
    if value >= 0.1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")

