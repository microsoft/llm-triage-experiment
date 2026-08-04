from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from . import constants as C


def _trapezoid(y, x) -> float:
    fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(fn(y, x))


def matched_pairs(
    belief: Mapping[str, float],
    decisions: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    contexts = [context for context in decisions if context in belief]
    return (
        np.asarray([belief[context] for context in contexts], dtype=float),
        np.asarray([decisions[context] for context in contexts], dtype=float),
    )


def scores_and_labels(
    belief: Mapping[str, float],
    gold: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    contexts = [context for context in belief if context in gold]
    return (
        np.asarray([belief[context] for context in contexts], dtype=float),
        np.asarray([gold[context] for context in contexts], dtype=int),
    )


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def fit_cost_function(beliefs: Sequence[float], decisions: Sequence[float]) -> dict:
    p = np.asarray(beliefs, dtype=float)
    a = np.asarray(decisions, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a)
    p = p[mask]
    a = a[mask]
    n = int(len(a))
    if n == 0:
        return {
            "n": 0,
            "c_fp": None,
            "c_fn": None,
            "ratio_fn_fp": None,
            "loglik": None,
            "frac_act": None,
            "degenerate": True,
            "threshold": None,
        }

    frac_act = float(a.mean())
    if frac_act in (0.0, 1.0):
        return {
            "n": n,
            "c_fp": None,
            "c_fn": None,
            "ratio_fn_fp": None,
            "loglik": None,
            "frac_act": frac_act,
            "degenerate": True,
            "threshold": None,
        }

    def neg_loglik(c: np.ndarray) -> float:
        c_fp, c_fn = c
        z = c_fn * p - c_fp * (1.0 - p)
        log_p_act = -np.logaddexp(0.0, -z)
        log_p_noact = -np.logaddexp(0.0, z)
        ll = a * log_p_act + (1.0 - a) * log_p_noact
        return -float(ll.sum())

    result = minimize(
        neg_loglik,
        x0=np.array([1.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (1e-6, None)],
    )
    c_fp, c_fn = float(result.x[0]), float(result.x[1])
    ratio = c_fn / c_fp if c_fp > 0 else None
    threshold = c_fp / (c_fp + c_fn) if (c_fp + c_fn) > 0 else None
    return {
        "n": n,
        "c_fp": c_fp,
        "c_fn": c_fn,
        "ratio_fn_fp": ratio,
        "loglik": -float(result.fun),
        "frac_act": frac_act,
        "degenerate": False,
        "threshold": threshold,
    }


def fit_cost_function_from_maps(
    belief: Mapping[str, float],
    decisions: Mapping[str, int],
) -> dict:
    p, d = matched_pairs(belief, decisions)
    return fit_cost_function(p, d)


def roc_curve(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray, float]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    y = y[order]
    threshold_indices = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tp = np.cumsum(y == 1)[threshold_indices]
    fp = np.cumsum(y == 0)[threshold_indices]
    tpr = np.concatenate([[0.0], tp / positives])
    fpr = np.concatenate([[0.0], fp / negatives])
    return fpr, tpr, _trapezoid(tpr, fpr)


def operating_point(decisions: Mapping[str, int], gold: Mapping[str, int]) -> dict:
    contexts = [context for context in decisions if context in gold]
    if not contexts:
        return {
            "n": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "tpr": float("nan"),
            "fpr": float("nan"),
            "accuracy": float("nan"),
        }
    pred = np.asarray([decisions[context] for context in contexts], dtype=int)
    y = np.asarray([gold[context] for context in contexts], dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    return {
        "n": len(contexts),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": float(tp / positives),
        "fpr": float(fp / negatives),
        "accuracy": float((pred == y).mean()),
    }


def best_fixed_threshold(
    belief: Mapping[str, float],
    gold: Mapping[str, int],
    ratio_fn_fp: float,
) -> dict:
    probabilities, labels = scores_and_labels(belief, gold)
    if len(labels) == 0:
        return {
            "threshold": float("nan"),
            "utility_ratio_fn_fp": float("nan"),
            "tpr": float("nan"),
            "fpr": float("nan"),
            "accuracy": float("nan"),
            "total_cost": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    candidates = np.unique(np.concatenate([[0.0], probabilities, [1.0 + 1e-9]]))
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    n = len(labels)
    best: dict | None = None
    best_cost = float("inf")
    c_fp = 1.0
    c_fn = float(ratio_fn_fp)
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        total_cost = c_fn * fn + c_fp * fp
        if total_cost < best_cost - 1e-12:
            best_cost = total_cost
            best = {
                "threshold": float(threshold),
                "utility_ratio_fn_fp": (
                    0.0
                    if threshold > 1.0
                    else float((1.0 - threshold) / threshold)
                    if threshold > 0
                    else float("inf")
                ),
                "tpr": float(tp / positives),
                "fpr": float(fp / negatives),
                "accuracy": float((tp + tn) / n),
                "total_cost": float(total_cost),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
    assert best is not None
    return best


def calibration_curve(
    belief: Mapping[str, float],
    gold: Mapping[str, int],
    n_bins: int = C.CALIBRATION_BINS,
) -> dict:
    probabilities, labels = scores_and_labels(belief, gold)
    if len(labels) == 0:
        raise ValueError("No shared observations for calibration")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.searchsorted(edges, probabilities, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_belief: list[float] = []
    event_rate: list[float] = []
    counts: list[int] = []
    for idx in range(n_bins):
        in_bin = bin_indices == idx
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_belief.append(float(probabilities[in_bin].mean()))
        event_rate.append(float(labels[in_bin].mean()))
        counts.append(count)

    counts_array = np.asarray(counts, dtype=float)
    mean_belief_array = np.asarray(mean_belief, dtype=float)
    event_rate_array = np.asarray(event_rate, dtype=float)
    ece = float(
        np.sum(counts_array * np.abs(event_rate_array - mean_belief_array)) / len(probabilities)
    )
    return {
        "mean_belief": mean_belief,
        "event_rate": event_rate,
        "counts": counts,
        "ece": ece,
    }


def consistency_indicators(
    belief: Mapping[str, float],
    decisions: Mapping[str, int],
    threshold: float | None,
) -> list[int]:
    if threshold is None or not np.isfinite(threshold):
        return []
    indicators: list[int] = []
    for context, decision in decisions.items():
        if context not in belief:
            continue
        implied = 1 if belief[context] >= threshold else 0
        indicators.append(int(implied == decision))
    return indicators


def bootstrap_ratio_ci(
    belief: Mapping[str, float],
    decisions: Mapping[str, int],
    n_boot: int = C.BOOTSTRAP_DRAWS,
    seed: int = C.BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    probabilities, actions = matched_pairs(belief, decisions)
    n = len(actions)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    recovered: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fit = fit_cost_function(probabilities[idx], actions[idx])
        ratio = fit.get("ratio_fn_fp")
        if not fit.get("degenerate") and ratio is not None and np.isfinite(ratio):
            recovered.append(float(ratio))
    if len(recovered) < max(10, n_boot // 20):
        return None
    lo, hi = np.percentile(recovered, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_prop_ci(
    indicators: Sequence[int],
    n_boot: int = C.BOOTSTRAP_DRAWS,
    seed: int = C.BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    values = np.asarray(indicators, dtype=float)
    n = len(values)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)
