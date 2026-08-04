from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from generic_RoC.generic_roc import _json_ready, best_fixed_utility, roc_curve


ROOT = Path(__file__).resolve().parents[1]


class GenericRocCliTests(unittest.TestCase):
    def test_auroc_is_invariant_to_tied_score_order(self) -> None:
        scores = np.asarray([0.8, 0.8, 0.2, 0.2])
        labels = np.asarray([1, 0, 1, 0])
        _fpr, _tpr, auc = roc_curve(scores, labels)
        _fpr, _tpr, reversed_auc = roc_curve(scores, labels[::-1])
        self.assertAlmostEqual(auc, 0.5)
        self.assertAlmostEqual(reversed_auc, auc)

    def test_always_negative_policy_has_zero_implied_ratio(self) -> None:
        result = best_fixed_utility([0.1, 0.2, 0.3], [0, 0, 0], 0.2)
        self.assertGreater(result.threshold, 1.0)
        self.assertEqual(result.utility_ratio_fn_fp, 0.0)

    def test_nonfinite_json_values_become_null(self) -> None:
        text = json.dumps(_json_ready({"nan": float("nan"), "inf": float("inf")}), allow_nan=False)
        self.assertEqual(json.loads(text), {"nan": None, "inf": None})

    def test_single_class_cli_writes_strict_json(self) -> None:
        output_dir = ROOT / "tests" / "artifacts" / "generic_roc_single_class"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "input.csv"
        input_path.write_text(
            "belief,label\n0.1,0\n0.2,0\n0.3,0\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "generic_RoC" / "generic_roc.py"),
                "--input",
                str(input_path),
                "--cost-ratio",
                "1:5",
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            check=True,
        )
        summary_text = (output_dir / "summary.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", summary_text)
        self.assertNotIn("Infinity", summary_text)
        summary = json.loads(summary_text)
        self.assertIsNone(summary["auroc"])
        self.assertEqual(summary["best_fixed_utility"]["utility_ratio_fn_fp"], 0.0)
        self.assertGreater((output_dir / "roc.png").stat().st_size, 0)

    def test_example_cli(self) -> None:
        output_dir = ROOT / "tests" / "artifacts" / "generic_roc_example"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            str(ROOT / "generic_RoC" / "generic_roc.py"),
            "--input",
            str(ROOT / "generic_RoC" / "example_dataset.csv"),
            "--decision-col",
            "decision",
            "--cost-ratio",
            "1",
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True)

        summary_path = output_dir / "summary.json"
        roc_path = output_dir / "roc.png"
        self.assertTrue(summary_path.exists())
        self.assertTrue(roc_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["n_rows_used"], 600)
        self.assertEqual(summary["target_cost_ratio_fn_fp"], 1.0)
        self.assertGreater(summary["auroc"], 0.75)


if __name__ == "__main__":
    unittest.main()
