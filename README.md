# Public offline triage analysis package

This package rebuilds the Nature Medicine triage analyses and figure set entirely offline from compact, analysis-ready inputs stored inside `public_github`.

## Contents

- `run_analysis.py` — one-command rebuild
- `analysis_summary.json` — generated summary of metrics and outputs
- `nature_medicine_paper\figures\` — generated manuscript figures
- `data\models\*.csv.gz` — compact model outputs with no raw responses
- `data\original_paper\DataExpanded_FINAL.csv.gz` — compact deployed-tool comparison data
- `generic_RoC\` — standalone generic ROC / utility tool
- `tests\` — focused validation tests

## Install

```bash
pip install -r requirements.txt
```

## Rebuild everything

```bash
python run_analysis.py
```

Generated outputs include:

- `fig1_emergency_triage_performance.png`
- `fig1_original_emergency_triage_performance.png`
- `fig1b_default_utilities.png`
- `fig2_recovered_capability.png`
- `fig3_roc_grid_gpt.png`
- `fig3_roc_grid_deepseek.png`
- `fig3_roc_grid_claude.png`
- `fig3_original_roc_grid_gpt.png`
- `fig3_original_roc_grid_deepseek.png`
- `fig3_original_roc_grid_claude.png`
- `decision_belief_consistency.png`
- `nature_decision_consistency.png`
- `fig4_recovered_threshold.png`
- `fig_belief_distributions.png`
- `fig_original_belief_distributions.png`
- `fig_calibration_curves.png`
- `fig_original_calibration_curves.png`
- `fig_prompt_context.png`
- `fig_prompt_belief.png`
- `fig_prompt_baseline.png`
- `fig_prompt_utility.png`
- `fig_prompt_threshold.png`

## Analysis conventions preserved

- Primary 576-case endpoint: original 960 variants excluding `C/D` cases; positives are `D`
- Expanded 1,248-case endpoint: all variants; positives are any gold label containing `D`
- Utility ratios: `0.01`, `0.1`, `0.2`, `1`, `5`, `10`, `100`
- Recovered utilities: discrete-choice logit fit
- Bootstrap: 500 draws, seed `0`
- Calibration: 10-bin ECE
- Utility-following summary: Spearman correlation
- Best fixed threshold summaries: cost-aware thresholding on elicited beliefs

`gpt-5.4-mini` is retained in `data\models` for completeness but is omitted from the 18 manuscript figure configurations referenced by `main.tex`.

The endpoint sizes describe the intended case sets. Failed or unparseable model
responses are excluded, matching the manuscript methods. Because availability
varies by model and regime, figures display model-specific denominators and
`analysis_summary.json` records expected, observed, and missing counts for every
ROC plus the denominator for each decision analysis. ROC operating points use
only cases with both a parsed belief and a parsed decision, so they never
include observations absent from the corresponding belief curve.

## Generic ROC tool

```bash
cd generic_RoC
python generic_roc.py --input example_dataset.csv --decision-col decision --cost-ratio 1
```

That command writes `roc.png` and `summary.json` beside the example input or inside `--output-dir`.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Verify the 5,000-line limit

```bash
python count_lines.py
```

This counts all `.py`, `.r`, `.js`, `.ts`, `.ps1`, and `.sh` files, including
the checker and tests, and exits unsuccessfully if the total reaches 5,000.

No license file is included because the source repository does not declare a
project license. Select and add the intended license before public release.

All code and runtime paths stay inside `public_github`.
