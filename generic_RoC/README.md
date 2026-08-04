# generic_RoC

Standalone offline ROC + utility recovery for labeled belief datasets.

## Run

```bash
cd generic_RoC
python generic_roc.py --input example_dataset.csv --decision-col decision --cost-ratio 1
```

Outputs:

- `summary.json`
- `roc.png`

## Input schema

Required columns:

- `belief`: probability in `[0, 1]`
- `label`: binary ground-truth outcome

Optional:

- `decision`: binary observed action

Binary fields accept `0/1`, `yes/no`, `true/false`, `positive/negative`, and similar tokens.

## Cost-ratio convention

Ratios are always **FN/FP**.

- `--cost-ratio 1` or `1:1` → balanced
- `--cost-ratio 10:1` → safety-leaning
- `--cost-ratio 1:5` → resource-leaning

## What it reports

1. AUROC of beliefs vs labels
2. Revealed FN/FP ratio implied by decisions (or labels if `decision` is omitted)
3. Best fixed utility for a target FN/FP objective

This tool is fully offline and depends only on `numpy`, `scipy`, and `matplotlib`.
