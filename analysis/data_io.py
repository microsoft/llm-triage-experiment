from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import constants as C


@dataclass(frozen=True)
class ContextMeta:
    context_id: str
    case_number: int
    gold_standard: str


@dataclass(frozen=True)
class OriginalPaperRow:
    case_id: str
    case_num: int
    gold_triage: str
    llm_triage: str
    race: str
    gender: str
    has_anchor: str
    has_barrier: str
    is_edge_case: str


@dataclass
class ConfigData:
    name: str
    family: str
    reasoning: str
    belief: dict[str, float] = field(default_factory=dict)
    decisions: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class StudyData:
    meta: dict[str, ContextMeta]
    configs: dict[str, ConfigData]
    original_paper: dict[str, OriginalPaperRow]
    source_rows: dict[str, int]


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, encoding="utf-8", newline="")


def _iter_rows(path: Path) -> Iterator[dict[str, str]]:
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _factorial_context_id(row: dict[str, str]) -> str:
    race_code = {"White": "W", "Black": "B"}[row["race"]]
    gender_code = {"man": "M", "woman": "W"}[row["gender"]]
    modifiers = ""
    if row["has_anchor"] == "yes":
        modifiers += "A"
    if row["has_barrier"] == "yes":
        modifiers += "X"
    suffix = f"-{modifiers}" if modifiers else ""
    return f"{row['case_id']}__{race_code}{gender_code}{suffix}"


def load_study_data(models_dir: Path = C.MODELS_DIR) -> StudyData:
    meta: dict[str, ContextMeta] = {}
    configs: dict[str, ConfigData] = {}
    source_rows: dict[str, int] = {}

    for model_path in sorted(models_dir.glob("*.csv.gz")):
        row_count = 0
        for row in _iter_rows(model_path):
            row_count += 1
            model = row["model"] or model_path.name.removesuffix(".csv.gz")
            reasoning = row["reasoning_effort"]
            config_name = f"{model}_re-{reasoning}"
            config = configs.setdefault(
                config_name,
                ConfigData(name=config_name, family=model, reasoning=reasoning),
            )

            context_id = row["context_id"]
            case_number = int(row["case_number"])
            gold_standard = row["gold_standard"]
            existing = meta.get(context_id)
            if existing is None:
                meta[context_id] = ContextMeta(
                    context_id=context_id,
                    case_number=case_number,
                    gold_standard=gold_standard,
                )
            elif (
                existing.case_number != case_number
                or existing.gold_standard != gold_standard
            ):
                raise ValueError(f"Inconsistent metadata for {context_id}")

            regime = row["regime"]
            probability = row.get("parsed_probability") or ""
            decision = row.get("parsed_decision") or ""
            if regime == "belief":
                if probability:
                    config.belief[context_id] = float(probability)
            elif decision:
                config.decisions.setdefault(regime, {})[context_id] = int(float(decision))
        source_rows[model_path.name] = row_count

    original_paper: dict[str, OriginalPaperRow] = {}
    original_rows = 0
    for row in _iter_rows(C.ORIGINAL_PAPER_FILE):
        original_rows += 1
        context_id = _factorial_context_id(row)
        original_paper[context_id] = OriginalPaperRow(
            case_id=context_id,
            case_num=int(row["case_num"]),
            gold_triage=row["gold_triage"],
            llm_triage=row["llm_triage"],
            race=row["race"],
            gender=row["gender"],
            has_anchor=row["has_anchor"],
            has_barrier=row["has_barrier"],
            is_edge_case=row["is_edge_case"],
        )
    source_rows[C.ORIGINAL_PAPER_FILE.name] = original_rows

    missing_original = sorted(set(meta) - set(original_paper))
    missing_model = sorted(set(original_paper) - set(meta))
    if missing_original or missing_model:
        raise ValueError(
            "Original-paper alignment mismatch: "
            f"{len(missing_original)} missing in original, "
            f"{len(missing_model)} missing in model data"
        )
    return StudyData(meta=meta, configs=configs, original_paper=original_paper, source_rows=source_rows)


def paper_config_names(study: StudyData) -> list[str]:
    return [name for name in C.PAPER_CONFIG_ORDER if name in study.configs]


def endpoint_gold(meta: dict[str, ContextMeta], endpoint: str) -> dict[str, int]:
    if endpoint == "primary":
        return {
            context_id: 1 if row.gold_standard == "D" else 0
            for context_id, row in meta.items()
            if row.case_number <= 30 and row.gold_standard != "C/D"
        }
    if endpoint == "expanded":
        return {
            context_id: 1 if "D" in row.gold_standard else 0
            for context_id, row in meta.items()
        }
    raise ValueError(f"Unknown endpoint: {endpoint}")


def endpoint_counts(meta: dict[str, ContextMeta], endpoint: str) -> dict[str, int]:
    gold = endpoint_gold(meta, endpoint)
    positives = int(sum(gold.values()))
    return {
        "n_contexts": len(gold),
        "n_positive": positives,
        "n_negative": len(gold) - positives,
    }


def original_paper_labels(rows: dict[str, OriginalPaperRow], endpoint: str) -> dict[str, int]:
    if endpoint == "primary":
        return {
            case_id: 1 if row.gold_triage == "D" else 0
            for case_id, row in rows.items()
            if row.case_num <= 60 and row.gold_triage != "C/D"
        }
    if endpoint == "expanded":
        return {case_id: 1 if "D" in row.gold_triage else 0 for case_id, row in rows.items()}
    raise ValueError(f"Unknown endpoint: {endpoint}")


def original_paper_decisions(rows: dict[str, OriginalPaperRow]) -> dict[str, int]:
    return {case_id: 1 if row.llm_triage == "D" else 0 for case_id, row in rows.items()}
