# Analysis-ready data

The seven compressed files in `models/` contain the successful parsed model
responses needed by the paper analysis. They include all reasoning levels and
both utility- and threshold-prompting experiments. Raw responses, prompts,
retry logs, API configuration, and credentials are intentionally excluded.

Model columns:

- `model`, `reasoning_effort`, `experiment`
- `context_id`, `case_number`, `gold_standard`
- `regime`, `c_fp`, `c_fn`, `p_star`
- `parsed_probability`, `parsed_decision`
- `their_triage` (the published ChatGPT Health decision for the matched case)

`original_paper/DataExpanded_FINAL.csv.gz` retains the compact columns needed
to validate and reproduce the matched ChatGPT Health comparison. All files are
read directly with Python's standard `csv` and `gzip` modules.
