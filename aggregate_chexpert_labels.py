"""
Aggregates CheXpert labels from individual reports back to sample-level and report-level structures.

Implements:
- Index & metadata alignment to match CheXpert output rows to original sample reports.
- Safe handling of empty/placeholder reports (assigned NaN across all categories).
- Hierarchical multi-report aggregation (1.0 > -1.0 > 0.0 > NaN) for sample-level labels.
- Generation of both sample-level label columns and granular per-report label lists.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from read_csv_data import load_csv

CHEXPERT_CATEGORIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

PATHOLOGY_CATEGORIES = [cat for cat in CHEXPERT_CATEGORIES if cat != "No Finding"]


def aggregate_values(values: List[Any]) -> Optional[float]:
    """
    Hierarchical multi-report pooling for a pathology:
    1.0 (Positive) > -1.0 (Uncertain) > 0.0 (Negative) > NaN (Unmentioned)
    """
    valid_vals = [v for v in values if pd.notna(v)]
    if not valid_vals:
        return np.nan
    if 1.0 in valid_vals or 1 in valid_vals:
        return 1.0
    if -1.0 in valid_vals or -1 in valid_vals:
        return -1.0
    if 0.0 in valid_vals or 0 in valid_vals:
        return 0.0
    return np.nan


def aggregate_labels(
    original_csv: Path,
    meta_json_path: Path,
    chexpert_output_csv: Path,
    output_labeled_csv: Path,
    export_flat_reports: bool = False,
    flat_reports_output: Optional[Path] = None,
    verbose: bool = True
):
    if verbose:
        print(f"[*] Loading metadata from: {meta_json_path}")
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    samples_meta = meta["samples_meta"]
    num_samples = len(samples_meta)

    if verbose:
        print(f"[*] Loading original dataset: {original_csv} (nrows={num_samples}) ...")
    orig_df = load_csv(original_csv, parse_lists=True, nrows=num_samples, verbose=False)

    if verbose:
        print(f"[*] Loading CheXpert output from: {chexpert_output_csv} ...")
    labeled_reports_df = pd.read_csv(chexpert_output_csv)

    # Ensure all CheXpert categories exist in labeled_reports_df
    for cat in CHEXPERT_CATEGORIES:
        if cat not in labeled_reports_df.columns:
            labeled_reports_df[cat] = np.nan

    # Build unique report id to labels mapping
    # Note: Row i in chexpert_output corresponds to unique_report_id = i
    unique_id_to_labels: Dict[int, Dict[str, float]] = {}
    for uid, row in labeled_reports_df.iterrows():
        unique_id_to_labels[uid] = {cat: row[cat] for cat in CHEXPERT_CATEGORIES}

    empty_report_labels = {cat: np.nan for cat in CHEXPERT_CATEGORIES}

    sample_level_labels: List[Dict[str, Any]] = []
    sample_report_labels: List[List[Dict[str, Any]]] = []
    flat_report_rows: List[Dict[str, Any]] = []

    for s_meta in samples_meta:
        sample_idx = s_meta["sample_idx"]
        subject_id = s_meta["subject_id"]
        uids = s_meta["unique_report_ids"]
        empty_flags = s_meta["empty_flags"]

        per_report_label_list = []
        pathology_pool: Dict[str, List[float]] = {cat: [] for cat in CHEXPERT_CATEGORIES}

        for r_idx, (uid, is_empty) in enumerate(zip(uids, empty_flags)):
            if is_empty or uid == -1 or uid not in unique_id_to_labels:
                r_labels = empty_report_labels.copy()
            else:
                r_labels = unique_id_to_labels[uid].copy()

            per_report_label_list.append(r_labels)

            for cat in CHEXPERT_CATEGORIES:
                val = r_labels.get(cat, np.nan)
                if pd.notna(val):
                    pathology_pool[cat].append(float(val))

            if export_flat_reports:
                flat_row = {
                    "sample_idx": sample_idx,
                    "subject_id": subject_id,
                    "report_idx": r_idx,
                    "is_empty": is_empty
                }
                flat_row.update(r_labels)
                flat_report_rows.append(flat_row)

        sample_report_labels.append(per_report_label_list)

        # Aggregate sample-level labels across reports
        agg_sample = {}
        any_positive_pathology = False
        for cat in PATHOLOGY_CATEGORIES:
            agg_val = aggregate_values(pathology_pool[cat])
            agg_sample[cat] = agg_val
            if agg_val == 1.0:
                any_positive_pathology = True

        # Determine "No Finding"
        if any_positive_pathology:
            agg_sample["No Finding"] = 0.0
        else:
            no_finding_raw = aggregate_values(pathology_pool["No Finding"])
            agg_sample["No Finding"] = no_finding_raw

        sample_level_labels.append(agg_sample)

    # Attach aggregated labels and report_labels to orig_df
    result_df = orig_df.copy()
    sample_labels_df = pd.DataFrame(sample_level_labels)

    for cat in CHEXPERT_CATEGORIES:
        result_df[cat] = sample_labels_df[cat].values

    # Convert np.nan to None for JSON serialization
    serialized_report_labels = []
    for rep_list in sample_report_labels:
        cleaned_rep_list = []
        for r_dict in rep_list:
            cleaned_rep_list.append({k: (None if pd.isna(v) else float(v)) for k, v in r_dict.items()})
        serialized_report_labels.append(json.dumps(cleaned_rep_list, ensure_ascii=False))

    result_df["report_labels"] = serialized_report_labels

    output_labeled_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_labeled_csv, index=False)

    if export_flat_reports and flat_reports_output:
        flat_reports_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flat_report_rows).to_csv(flat_reports_output, index=False)
        if verbose:
            print(f"[OK] Exported flat report labels to: {flat_reports_output}")

    if verbose:
        print("\n================ AGGREGATION SUMMARY ================")
        print(f"Total samples labeled    : {len(result_df)}")
        print(f"Output labeled file      : {output_labeled_csv}")
        print("Positive cases per category across samples:")
        for cat in CHEXPERT_CATEGORIES:
            pos_count = (result_df[cat] == 1.0).sum()
            print(f"  - {cat:28s}: {pos_count} samples ({round(pos_count/len(result_df)*100, 1)}%)")
        print("====================================================\n")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate CheXpert labels to sample level")
    parser.add_argument("--original_csv", type=str, default="mimic_cxr_aug_train.csv")
    parser.add_argument("--meta", type=str, default="data_work/mapping_metadata.json")
    parser.add_argument("--chexpert_output", type=str, default="data_work/chexpert_output.csv")
    parser.add_argument("--output_csv", type=str, default="data_work/mimic_cxr_aug_train_labeled.csv")
    parser.add_argument("--export_flat", action="store_true", help="Also export flat report-level table")
    parser.add_argument("--flat_output", type=str, default="data_work/reports_labeled_flat.csv")

    args = parser.parse_args()
    aggregate_labels(
        original_csv=Path(args.original_csv),
        meta_json_path=Path(args.meta),
        chexpert_output_csv=Path(args.chexpert_output),
        output_labeled_csv=Path(args.output_csv),
        export_flat_reports=args.export_flat,
        flat_reports_output=Path(args.flat_output) if args.flat_output else None,
        verbose=True
    )
