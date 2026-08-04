#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


TRUE_TOKENS = {"1", "1.0", "yes", "y", "true", "t", "positive", "pos", "admit"}
FALSE_TOKENS = {"0", "0.0", "no", "n", "false", "f", "negative", "neg", "discharge"}


def parse_cost_ratio(value: str) -> float:
    text = str(value).strip()
    if ":" in text:
        left, right = text.split(":", 1)
        fn = float(left)
        fp = float(right)
        if fn <= 0 or fp <= 0:
            raise ValueError("Cost ratios must be positive")
        return fn / fp
    ratio = float(text)
    if ratio <= 0:
        raise ValueError("Cost ratio must be positive")
    return ratio


def _to_binary(raw: str | None, field_name: str) -> int | None:
    if raw is None:
        return None
    token = str(raw).strip().lower()
    if token == "":
        return None
    if token in TRUE_TOKENS:
        return 1
    if token in FALSE_TOKENS:
        return 0
    number = float(token)
    if number == 1:
        return 1
    if number == 0:
        return 0
    raise ValueError(f"{field_name}={raw!r} is not binary")


@dataclass
class Dataset:
    beliefs: np.ndarray
    labels: np.ndarray
    decisions: np.ndarray | None
    n_rows_read: int
    n_rows_used: int


def load_dataset(path: Path, belief_col: str, label_col: str, decision_col: str | None) -> Dataset:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        required = [belief_col, label_col] + ([decision_col] if decision_col else [])
        missing = [name for name in required if name and name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        beliefs: list[float] = []
        labels: list[int] = []
        decisions: list[float] = []
        n_rows_read = 0
        for row in reader:
            n_rows_read += 1
            raw_belief = (row.get(belief_col) or "").strip()
            raw_label = row.get(label_col)
            if raw_belief == "":
                continue
            label = _to_binary(raw_label, label_col)
            if label is None:
                continue
            belief = float(raw_belief)
            if not 0.0 <= belief <= 1.0:
                raise ValueError(f"{belief_col} must be between 0 and 1")
            beliefs.append(belief)
            labels.append(label)
            if decision_col:
                decision = _to_binary(row.get(decision_col), decision_col)
                decisions.append(float("nan") if decision is None else float(decision))
    if not beliefs:
        raise ValueError("No usable rows found")
    return Dataset(
        beliefs=np.asarray(beliefs, dtype=float),
        labels=np.asarray(labels, dtype=int),
        decisions=np.asarray(decisions, dtype=float) if decision_col else None,
        n_rows_read=n_rows_read,
        n_rows_used=len(beliefs),
    )


def roc_curve(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    labels = labels[order]
    threshold_indices = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(labels) - 1]
    tp = np.cumsum(labels == 1)[threshold_indices]
    fp = np.cumsum(labels == 0)[threshold_indices]
    tpr = np.concatenate([[0.0], tp / positives])
    fpr = np.concatenate([[0.0], fp / negatives])
    area_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return fpr, tpr, float(area_fn(tpr, fpr))


def fit_cost_function(beliefs, decisions) -> dict:
    beliefs = np.asarray(beliefs, dtype=float)
    decisions = np.asarray(decisions, dtype=float)
    mask = np.isfinite(beliefs) & np.isfinite(decisions)
    beliefs = beliefs[mask]
    decisions = decisions[mask]
    n = int(len(decisions))
    if n == 0:
        return {"n": 0, "c_fp": None, "c_fn": None, "ratio_fn_fp": None, "loglik": None, "frac_act": None, "degenerate": True}
    frac_act = float(decisions.mean())
    if frac_act in (0.0, 1.0):
        return {"n": n, "c_fp": None, "c_fn": None, "ratio_fn_fp": None, "loglik": None, "frac_act": frac_act, "degenerate": True}

    def neg_loglik(params: np.ndarray) -> float:
        c_fp, c_fn = params
        z = c_fn * beliefs - c_fp * (1.0 - beliefs)
        log_p_act = -np.logaddexp(0.0, -z)
        log_p_noact = -np.logaddexp(0.0, z)
        return -float((decisions * log_p_act + (1.0 - decisions) * log_p_noact).sum())

    result = minimize(
        neg_loglik,
        x0=np.array([1.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None)],
    )
    c_fp, c_fn = float(result.x[0]), float(result.x[1])
    return {
        "n": n,
        "c_fp": c_fp,
        "c_fn": c_fn,
        "ratio_fn_fp": c_fn / c_fp if c_fp > 0 else None,
        "loglik": -float(result.fun),
        "frac_act": frac_act,
        "degenerate": False,
    }


def operating_point(decisions, labels):
    decisions = np.asarray(decisions, dtype=float)
    labels = np.asarray(labels, dtype=int)
    mask = np.isfinite(decisions)
    decisions = decisions[mask].astype(int)
    labels = labels[mask]
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    return {
        "tpr": float(((decisions == 1) & (labels == 1)).sum() / positives),
        "fpr": float(((decisions == 1) & (labels == 0)).sum() / negatives),
        "accuracy": float((decisions == labels).mean()) if len(labels) else float("nan"),
    }


@dataclass
class BestFixedUtility:
    cost_ratio_fn_fp: float
    threshold: float
    utility_ratio_fn_fp: float
    tpr: float
    fpr: float
    accuracy: float
    total_cost: float
    tp: int
    fp: int
    tn: int
    fn: int


def best_fixed_utility(beliefs, labels, cost_ratio: float) -> BestFixedUtility:
    beliefs = np.asarray(beliefs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    candidates = np.unique(np.concatenate([[0.0], beliefs, [1.0 + 1e-9]]))
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    best_cost = float("inf")
    best: BestFixedUtility | None = None
    for threshold in candidates:
        pred = (beliefs >= threshold).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        total_cost = fp + cost_ratio * fn
        if total_cost < best_cost - 1e-12:
            best_cost = total_cost
            best = BestFixedUtility(
                cost_ratio_fn_fp=float(cost_ratio),
                threshold=float(threshold),
                utility_ratio_fn_fp=(
                    0.0
                    if threshold > 1.0
                    else float((1.0 - threshold) / threshold)
                    if threshold > 0
                    else float("inf")
                ),
                tpr=float(tp / positives),
                fpr=float(fp / negatives),
                accuracy=float((tp + tn) / len(labels)),
                total_cost=float(total_cost),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            )
    assert best is not None
    return best


def _ratio_text(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "inf"
    if value == 0:
        return "0:1"
    if value >= 1:
        return f"{value:.3g}:1"
    return f"1:{(1.0 / value):.3g}"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def analyze(dataset: Dataset, cost_ratio: float) -> dict:
    fpr, tpr, auc = roc_curve(dataset.beliefs, dataset.labels)
    if dataset.decisions is not None and np.isfinite(dataset.decisions).any():
        implied = fit_cost_function(dataset.beliefs, dataset.decisions)
        implied["source"] = "decisions"
    else:
        implied = fit_cost_function(dataset.beliefs, dataset.labels.astype(float))
        implied["source"] = "labels"
    best = best_fixed_utility(dataset.beliefs, dataset.labels, cost_ratio)
    return {
        "n_rows_read": dataset.n_rows_read,
        "n_rows_used": dataset.n_rows_used,
        "n_positive": int((dataset.labels == 1).sum()),
        "n_negative": int((dataset.labels == 0).sum()),
        "auroc": auc,
        "implied_ratio": implied,
        "target_cost_ratio_fn_fp": cost_ratio,
        "best_fixed_utility": asdict(best),
    }


def plot_roc(dataset: Dataset, results: dict, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, auc = roc_curve(dataset.beliefs, dataset.labels)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot(fpr, tpr, color="0.45", lw=2.6, label=f"Belief ROC (AUROC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Chance")
    if dataset.decisions is not None and np.isfinite(dataset.decisions).any():
        point = operating_point(dataset.decisions, dataset.labels)
        label = "Observed decisions"
        ratio = results["implied_ratio"].get("ratio_fn_fp")
        if ratio is not None:
            label += f" (implied FN/FP={ratio:.2f})"
        ax.plot(point["fpr"], point["tpr"], "o", color="#3182BD", markeredgecolor="black", markeredgewidth=0.8, markersize=12, label=label)
    best = results["best_fixed_utility"]
    ax.plot(
        best["fpr"],
        best["tpr"],
        "s",
        color="gold",
        markeredgecolor="black",
        markeredgewidth=1.0,
        markersize=16,
        label=f"Best Fixed Utility (prompt FN/FP={_ratio_text(best['utility_ratio_fn_fp'])}, thr={best['threshold']:.2f})",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.93)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_report(results: dict) -> None:
    implied = results["implied_ratio"]
    best = results["best_fixed_utility"]
    print("=" * 68)
    print("Generic ROC / utility analysis")
    print("=" * 68)
    print(f"rows used           : {results['n_rows_used']} of {results['n_rows_read']} ({results['n_positive']} positive, {results['n_negative']} negative)")
    print(f"AUROC               : {results['auroc']:.3f}")
    if implied.get("degenerate"):
        print(f"implied FN/FP       : degenerate ({implied.get('source')})")
    else:
        print(
            "implied FN/FP       : "
            f"{implied['ratio_fn_fp']:.3f} ({_ratio_text(implied['ratio_fn_fp'])}) "
            f"from {implied['source']}"
        )
    print(
        f"target FN/FP        : {results['target_cost_ratio_fn_fp']:.3g} "
        f"({_ratio_text(results['target_cost_ratio_fn_fp'])})"
    )
    print(
        "best fixed utility  : "
        f"thr={best['threshold']:.3f}, TPR={best['tpr']:.3f}, FPR={best['fpr']:.3f}, "
        f"prompt FN/FP={_ratio_text(best['utility_ratio_fn_fp'])}"
    )
    print("=" * 68)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a belief ROC, recover the implied FN/FP ratio, and find the best fixed utility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--belief-col", default="belief")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--decision-col", default=None)
    parser.add_argument("--cost-ratio", default="1")
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--title", default="Belief ROC")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_dataset(args.input, args.belief_col, args.label_col, args.decision_col)
    results = analyze(dataset, parse_cost_ratio(args.cost_ratio))
    print_report(results)
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(results), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")
    if not args.no_plot:
        plot_path = output_dir / "roc.png"
        plot_roc(dataset, results, plot_path, args.title)
        print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
