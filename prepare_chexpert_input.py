"""
Prepares CheXpert labeler input from MIMIC-CXR augmented datasets.

Handles:
- Extracting list of N reports from each sample.
- Filtering out empty/placeholder reports.
- Deduplicating reports to dramatically reduce CheXpert labeling compute time.
- Generating headerless single-column CSV formatted for CheXpert-labeler.
- Saving metadata mapping to easily restore and aggregate labels back to samples.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional
import pandas as pd

from read_csv_data import load_csv


def is_placeholder_report(text: str) -> bool:
    """Check if report is empty or trivial placeholder (e.g. 'Findings:  Impression:')."""
    if not isinstance(text, str):
        return True
    cleaned = text.strip().lower()
    if not cleaned:
        return True
    # Placeholders common in MIMIC-CXR
    trivial_variants = {
        "findings: impression:",
        "findings:  impression:",
        "findings: impression",
        "findings:",
        "impression:",
        "findings: none. impression: none.",
        "findings: none. impression: none",
        "findings: none. impression:",
    }
    return cleaned in trivial_variants


def prepare_input(
    input_path: Path,
    output_dir: Path,
    column: str = "text",
    nrows: Optional[int] = None,
    verbose: bool = True
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[*] Reading dataset: {input_path} (nrows={nrows}) ...")
    df = load_csv(input_path, parse_lists=True, nrows=nrows, verbose=verbose)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset. Available columns: {df.columns.tolist()}")

    unique_report_to_id = {}
    unique_reports_list = []
    
    samples_meta = []
    total_reports_count = 0
    empty_reports_count = 0

    for sample_idx, row in df.iterrows():
        subject_id = row.get("subject_id")
        reports = row[column]
        if not isinstance(reports, list):
            reports = [reports] if pd.notna(reports) else []

        sample_unique_ids = []
        sample_empty_flags = []

        for report_idx, report in enumerate(reports):
            total_reports_count += 1
            if not isinstance(report, str) or is_placeholder_report(report):
                empty_reports_count += 1
                sample_empty_flags.append(True)
                sample_unique_ids.append(-1)  # -1 represents empty/placeholder
            else:
                sample_empty_flags.append(False)
                clean_text = " ".join(report.split())  # normalize whitespace
                if clean_text not in unique_report_to_id:
                    uid = len(unique_reports_list)
                    unique_report_to_id[clean_text] = uid
                    unique_reports_list.append(clean_text)
                else:
                    uid = unique_report_to_id[clean_text]
                sample_unique_ids.append(uid)

        samples_meta.append({
            "sample_idx": int(sample_idx),
            "subject_id": int(subject_id) if pd.notna(subject_id) else None,
            "num_reports": len(reports),
            "unique_report_ids": sample_unique_ids,
            "empty_flags": sample_empty_flags
        })

    # Export headerless, single-column CSV for CheXpert
    chexpert_input_csv = output_dir / "chexpert_input.csv"
    with open(chexpert_input_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for text in unique_reports_list:
            writer.writerow([text])

    # Save mapping metadata
    metadata = {
        "input_file": str(input_path.name),
        "column": column,
        "num_samples": len(df),
        "total_reports": total_reports_count,
        "unique_reports": len(unique_reports_list),
        "empty_reports": empty_reports_count,
        "samples_meta": samples_meta,
        "unique_reports": unique_reports_list
    }
    
    meta_json_path = output_dir / "mapping_metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if verbose:
        print("\n================ PREPARATION SUMMARY ================")
        print(f"Total samples processed : {len(df)}")
        print(f"Total reports found     : {total_reports_count}")
        print(f"Empty/trivial reports   : {empty_reports_count} (skipped from model)")
        print(f"Unique reports to label : {len(unique_reports_list)} (saved ~{round((1 - len(unique_reports_list)/max(1, total_reports_count - empty_reports_count))*100, 1)}% redundant parsing)")
        print(f"CheXpert input CSV      : {chexpert_input_csv}")
        print(f"Mapping metadata JSON   : {meta_json_path}")
        print("====================================================\n")

    return chexpert_input_csv, meta_json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare reports from MIMIC-CXR for CheXpert labeler")
    parser.add_argument("--input", type=str, default="mimic_cxr_aug_train.csv", help="Path to input CSV")
    parser.add_argument("--output_dir", type=str, default="data_work", help="Directory to save prepared files")
    parser.add_argument("--column", type=str, default="text", help="Column containing report list")
    parser.add_argument("--nrows", type=int, default=None, help="Number of rows to process (for testing)")
    
    args = parser.parse_args()
    prepare_input(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        column=args.column,
        nrows=args.nrows,
        verbose=True
    )
