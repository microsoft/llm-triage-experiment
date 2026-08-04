from __future__ import annotations

import unittest

import numpy as np

from analysis.data_io import endpoint_gold, load_study_data
from analysis.metrics import best_fixed_threshold, calibration_curve, fit_cost_function_from_maps, roc_curve, scores_and_labels


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = load_study_data()
        cls.config = cls.study.configs["gpt-5-mini_re-high"]

    def test_default_fit_is_finite(self) -> None:
        fit = fit_cost_function_from_maps(self.config.belief, self.config.decisions["decision_baseline"])
        self.assertFalse(fit["degenerate"])
        self.assertGreater(fit["ratio_fn_fp"], 0)
        self.assertGreaterEqual(fit["threshold"], 0)
        self.assertLessEqual(fit["threshold"], 1)

    def test_primary_auroc_is_high(self) -> None:
        gold = endpoint_gold(self.study.meta, "primary")
        scores, labels = scores_and_labels(self.config.belief, gold)
        _fpr, _tpr, auc = roc_curve(scores, labels)
        self.assertGreater(auc, 0.9)

    def test_auroc_is_invariant_to_tied_score_order(self) -> None:
        scores = np.asarray([0.8, 0.8, 0.2, 0.2])
        labels = np.asarray([1, 0, 1, 0])
        _fpr, _tpr, auc = roc_curve(scores, labels)
        _fpr, _tpr, reversed_auc = roc_curve(scores, labels[::-1])
        self.assertAlmostEqual(auc, 0.5)
        self.assertAlmostEqual(reversed_auc, auc)

    def test_safety_threshold_is_lower(self) -> None:
        gold = endpoint_gold(self.study.meta, "expanded")
        balanced = best_fixed_threshold(self.config.belief, gold, 1.0)
        safety = best_fixed_threshold(self.config.belief, gold, 5.0)
        self.assertLessEqual(safety["threshold"], balanced["threshold"])

    def test_always_negative_policy_has_zero_implied_ratio(self) -> None:
        result = best_fixed_threshold(
            {"a": 0.1, "b": 0.2, "c": 0.3},
            {"a": 0, "b": 0, "c": 0},
            0.2,
        )
        self.assertGreater(result["threshold"], 1.0)
        self.assertEqual(result["utility_ratio_fn_fp"], 0.0)

    def test_calibration_counts_cover_endpoint(self) -> None:
        gold = endpoint_gold(self.study.meta, "primary")
        calibration = calibration_curve(self.config.belief, gold)
        self.assertEqual(sum(calibration["counts"]), 576)


if __name__ == "__main__":
    unittest.main()
