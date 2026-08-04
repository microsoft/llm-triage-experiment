from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis import FIGURES_DIR, SUMMARY_PATH, load_study_data
from analysis import constants as C
from analysis.data_io import endpoint_counts, endpoint_gold, original_paper_decisions, original_paper_labels
from analysis.figures import (
    build_belief_distributions,
    build_calibration,
    build_decision_consistency,
    build_default_utilities,
    build_main_roc,
    build_nature_consistency,
    build_recovered_capability,
    build_recovered_threshold,
    build_roc_grid,
)
from analysis.metrics import (
    best_fixed_threshold,
    bootstrap_prop_ci,
    bootstrap_ratio_ci,
    calibration_curve,
    consistency_indicators,
    fit_cost_function_from_maps,
    operating_point,
    roc_curve,
    scores_and_labels,
    spearman,
)
from analysis.prompts import build_prompt_figures


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_summary() -> dict:
    study = load_study_data()
    original_decisions = original_paper_decisions(study.original_paper)
    summary = {
        "metadata": {
            "bootstrap_draws": C.BOOTSTRAP_DRAWS,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
            "calibration_bins": C.CALIBRATION_BINS,
            "response_handling": "Unanswered or unparseable model responses are excluded per analysis; denominators are reported below.",
            "utility_costs": [{"c_fp": fp, "c_fn": fn, "ratio_fn_fp": fn / fp} for fp, fn in C.UTILITY_COSTS],
            "source_rows": study.source_rows,
        },
        "endpoints": {
            "primary": endpoint_counts(study.meta, "primary"),
            "expanded": endpoint_counts(study.meta, "expanded"),
        },
        "configs": {},
        "chatgpt_health": {},
    }

    gold_by_endpoint = {
        endpoint: endpoint_gold(study.meta, endpoint) for endpoint in ("primary", "expanded")
    }
    for endpoint in ("primary", "expanded"):
        gold = original_paper_labels(study.original_paper, endpoint)
        decisions = {context: original_decisions[context] for context in gold}
        summary["chatgpt_health"][endpoint] = operating_point(decisions, gold)

    for name, config in study.configs.items():
        baseline_decisions = config.decisions.get("decision_baseline", {})
        baseline_fit = fit_cost_function_from_maps(config.belief, baseline_decisions)
        recovered_by_regime: dict[str, float | None] = {}
        threshold_following: dict[str, dict[str, float | None]] = {}
        est_consistency: dict[str, float | None] = {}
        est_consistency_n: dict[str, int] = {}
        est_consistency_ci: dict[str, list[float] | None] = {}
        nature_agreement: dict[str, float | None] = {}
        nature_agreement_n: dict[str, int] = {}
        nature_agreement_ci: dict[str, list[float] | None] = {}

        target_logs: list[float] = []
        recovered_logs: list[float] = []

        all_fit_records = {"decision_baseline": baseline_fit}
        for regime in C.UTILITY_REGIMES:
            fit = fit_cost_function_from_maps(config.belief, config.decisions.get(regime, {}))
            all_fit_records[regime] = fit
            recovered_by_regime[regime] = fit["ratio_fn_fp"]
            if fit["ratio_fn_fp"] is not None:
                target_logs.append(np.log(C.TARGET_RATIOS[regime]))
                recovered_logs.append(np.log(fit["ratio_fn_fp"]))

        for regime in ["decision_baseline", *C.UTILITY_REGIMES]:
            fit = all_fit_records[regime]
            indicators = consistency_indicators(config.belief, config.decisions.get(regime, {}), fit["threshold"])
            est_consistency[regime] = float(np.mean(indicators)) if indicators else None
            est_consistency_n[regime] = len(indicators)
            ci = bootstrap_prop_ci(indicators) if indicators else None
            est_consistency_ci[regime] = list(ci) if ci is not None else None

            decision_map = config.decisions.get(regime, {})
            contexts = [context for context in decision_map if context in original_decisions]
            if contexts:
                matches = [int(decision_map[context] == original_decisions[context]) for context in contexts]
                nature_agreement[regime] = float(np.mean(matches))
                nature_agreement_n[regime] = len(matches)
                ci = bootstrap_prop_ci(matches)
                nature_agreement_ci[regime] = list(ci) if ci is not None else None
            else:
                nature_agreement[regime] = None
                nature_agreement_n[regime] = 0
                nature_agreement_ci[regime] = None

        for regime in C.THRESHOLD_REGIMES:
            fit = fit_cost_function_from_maps(config.belief, config.decisions.get(regime, {}))
            threshold_following[regime] = {
                "prompted_threshold": C.PROMPTED_THRESHOLDS[regime],
                "recovered_threshold": fit["threshold"],
                "recovered_ratio_fn_fp": fit["ratio_fn_fp"],
                "n": fit["n"],
            }

        endpoint_data = {}
        for endpoint, gold in gold_by_endpoint.items():
            scores, labels = scores_and_labels(config.belief, gold)
            fpr, tpr, auc = roc_curve(scores, labels)
            operating_points = {}
            for regime in ["decision_baseline", *C.UTILITY_REGIMES]:
                decision_map = config.decisions.get(regime, {})
                matched_gold = {
                    context: gold[context]
                    for context in decision_map
                    if context in config.belief and context in gold
                }
                operating_points[regime] = operating_point(decision_map, matched_gold)
            best_fixed = {
                C.ratio_key(ratio): best_fixed_threshold(config.belief, gold, ratio)
                for ratio in C.FIXED_UTILITY_RATIOS
            }
            endpoint_data[endpoint] = {
                "coverage": {
                    "expected_n": len(gold),
                    "observed_n": len(labels),
                    "missing_n": len(gold) - len(labels),
                    "observed_positive": int(labels.sum()),
                    "observed_negative": int(len(labels) - labels.sum()),
                },
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc, "n": len(labels)},
                "operating_points": operating_points,
                "best_fixed": best_fixed,
                "calibration": calibration_curve(config.belief, gold),
            }

        summary["configs"][name] = {
            "family": config.family,
            "reasoning": config.reasoning,
            "paper_config": name in C.PAPER_CONFIG_ORDER,
            "baseline_fit": baseline_fit,
            "baseline_ratio_ci": None,
            "recovered_by_regime": recovered_by_regime,
            "spearman_recovered_utilities": spearman(target_logs, recovered_logs) if len(recovered_logs) >= 3 else None,
            "est_consistency": est_consistency,
            "est_consistency_n": est_consistency_n,
            "est_consistency_ci": est_consistency_ci,
            "nature_agreement": nature_agreement,
            "nature_agreement_n": nature_agreement_n,
            "nature_agreement_ci": nature_agreement_ci,
            "threshold_following": threshold_following,
            "endpoints": endpoint_data,
        }
        baseline_ci = bootstrap_ratio_ci(config.belief, baseline_decisions) if baseline_decisions else None
        if baseline_ci is not None:
            summary["configs"][name]["baseline_ratio_ci"] = list(baseline_ci)
    return {"study": study, "summary": summary}


def write_outputs(study, summary: dict) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    build_main_roc(study, summary, "primary", FIGURES_DIR / "fig1_emergency_triage_performance.png")
    generated.append("fig1_emergency_triage_performance.png")
    build_main_roc(study, summary, "expanded", FIGURES_DIR / "fig1_original_emergency_triage_performance.png")
    generated.append("fig1_original_emergency_triage_performance.png")
    build_default_utilities(summary, FIGURES_DIR / "fig1b_default_utilities.png")
    generated.append("fig1b_default_utilities.png")
    build_recovered_capability(summary, FIGURES_DIR / "fig2_recovered_capability.png")
    generated.append("fig2_recovered_capability.png")
    for provider_slug in ("gpt", "deepseek", "claude"):
        build_roc_grid(study, summary, "primary", provider_slug, FIGURES_DIR / f"fig3_roc_grid_{provider_slug}.png")
        generated.append(f"fig3_roc_grid_{provider_slug}.png")
        build_roc_grid(study, summary, "expanded", provider_slug, FIGURES_DIR / f"fig3_original_roc_grid_{provider_slug}.png")
        generated.append(f"fig3_original_roc_grid_{provider_slug}.png")
    build_decision_consistency(summary, FIGURES_DIR / "decision_belief_consistency.png")
    generated.append("decision_belief_consistency.png")
    build_nature_consistency(summary, FIGURES_DIR / "nature_decision_consistency.png")
    generated.append("nature_decision_consistency.png")
    build_recovered_threshold(summary, FIGURES_DIR / "fig4_recovered_threshold.png")
    generated.append("fig4_recovered_threshold.png")
    build_belief_distributions(study, summary, "primary", FIGURES_DIR / "fig_belief_distributions.png")
    generated.append("fig_belief_distributions.png")
    build_belief_distributions(study, summary, "expanded", FIGURES_DIR / "fig_original_belief_distributions.png")
    generated.append("fig_original_belief_distributions.png")
    build_calibration(summary, "primary", FIGURES_DIR / "fig_calibration_curves.png")
    generated.append("fig_calibration_curves.png")
    build_calibration(summary, "expanded", FIGURES_DIR / "fig_original_calibration_curves.png")
    generated.append("fig_original_calibration_curves.png")
    generated.extend(build_prompt_figures(FIGURES_DIR))
    return generated


def main() -> int:
    built = build_summary()
    study = built["study"]
    summary = built["summary"]
    generated = write_outputs(study, summary)
    summary["metadata"]["generated_figures"] = generated
    summary["metadata"]["figure_sizes_bytes"] = {
        filename: (FIGURES_DIR / filename).stat().st_size for filename in generated
    }
    SUMMARY_PATH.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {len(generated)} figures to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
