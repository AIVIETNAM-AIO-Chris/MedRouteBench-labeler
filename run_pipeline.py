"""
End-to-end pipeline orchestrator for MIMIC-CXR CheXpert labeling.

Usage:
  # 1. Chạy bước chuẩn bị dữ liệu (test 20 dòng):
  python run_pipeline.py --step prepare --nrows 20

  # 2. Chạy dán nhãn bằng Docker:
  python run_pipeline.py --step label

  # 3. Chạy bước tổng hợp nhãn:
  python run_pipeline.py --step aggregate

  # 4. Chạy toàn bộ (end-to-end) trên tập test:
  python run_pipeline.py --step all --nrows 20
"""

import argparse
import subprocess
import sys
from pathlib import Path

from prepare_chexpert_input import prepare_input
from aggregate_chexpert_labels import aggregate_labels


def run_docker_labeler(
    input_csv_rel: str,
    output_csv_rel: str,
    workspace_dir: Path
):
    """Run chexpert-labeler inside Docker container."""
    # Ensure paths inside container use forward slashes under /data
    input_container = f"/data/{Path(input_csv_rel).as_posix()}"
    output_container = f"/data/{Path(output_csv_rel).as_posix()}"

    workspace_str = str(workspace_dir.resolve()).replace("\\", "/")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace_str}:/data",
        "chexpert-labeler:latest",
        "python", "label.py",
        "--reports_path", input_container,
        "--output_path", output_container,
        "--verbose"
    ]

    print("\n[*] Executing Docker CheXpert Labeler:")
    print(" ".join(cmd))
    print("------------------------------------------------------------")

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[!] Docker run failed with exit code {res.returncode}")
        print("[!] Hay dam bao Docker Desktop dang chay va image 'chexpert-labeler:latest' da duoc build.")
        sys.exit(res.returncode)
    print("------------------------------------------------------------")
    print(f"[OK] CheXpert labeler finished successfully -> {output_csv_rel}\n")


def run_local_labeler(
    input_csv: Path,
    output_csv: Path,
    labeler_dir: Path
):
    """Run chexpert-labeler directly on the host using local Python."""
    import os
    env = os.environ.copy()
    negbio_dir = labeler_dir / "NegBio"
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{negbio_dir}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(negbio_dir)

    cmd = [
        sys.executable,
        str(labeler_dir / "label.py"),
        "--reports_path", str(input_csv.resolve()),
        "--output_path", str(output_csv.resolve()),
        "--verbose"
    ]

    print("\n[*] Executing Local CheXpert Labeler (NO DOCKER):")
    print(" ".join(cmd))
    print("------------------------------------------------------------")

    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        print(f"\n[!] Local CheXpert run failed with exit code {res.returncode}")
        print("[!] Hay dam bao ban da cai dat moi truong Python 3.7 + Java JRE + NegBio day du.")
        sys.exit(res.returncode)
    print("------------------------------------------------------------")
    print(f"[OK] CheXpert labeler finished successfully -> {output_csv}\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-end CheXpert labeling pipeline for MIMIC-CXR")
    parser.add_argument("--step", choices=["prepare", "label", "aggregate", "all"], default="prepare",
                        help="Pipeline step to run: prepare, label, aggregate, or all")
    parser.add_argument("--no_docker", action="store_true",
                        help="Run labeler directly using local Python without Docker")
    parser.add_argument("--input", type=str, default="mimic_cxr_aug_train.csv",
                        help="Input MIMIC-CXR CSV file")
    parser.add_argument("--work_dir", type=str, default="data_work",
                        help="Working directory for intermediate and output files")
    parser.add_argument("--column", type=str, default="text",
                        help="Column containing report lists ('text' or 'text_augment')")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Number of rows to process (default: None for full dataset)")
    parser.add_argument("--output_labeled", type=str, default=None,
                        help="Final output labeled CSV path (defaults to <work_dir>/mimic_cxr_aug_train_labeled.csv)")

    args = parser.parse_args()

    workspace_dir = Path(__file__).parent.resolve()
    work_dir = workspace_dir / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    input_csv = workspace_dir / args.input
    chexpert_input_csv = work_dir / "chexpert_input.csv"
    meta_json = work_dir / "mapping_metadata.json"
    chexpert_output_csv = work_dir / "chexpert_output.csv"
    
    if args.output_labeled:
        final_output_csv = Path(args.output_labeled)
    else:
        final_output_csv = work_dir / "mimic_cxr_aug_train_labeled.csv"

    # Step: Prepare
    if args.step in ["prepare", "all"]:
        print("\n>>> STEP 1: PREPARING DATA FOR CHEXPERT LABELER <<<")
        prepare_input(
            input_path=input_csv,
            output_dir=work_dir,
            column=args.column,
            nrows=args.nrows,
            verbose=True
        )

    # Step: Label (Docker or Local Python)
    if args.step in ["label", "all"]:
        if not chexpert_input_csv.exists():
            print(f"[!] File not found: {chexpert_input_csv}. Hay chay step 'prepare' truoc!")
            sys.exit(1)

        if args.no_docker:
            print("\n>>> STEP 2: RUNNING CHEXPERT LABELER VIA LOCAL PYTHON (NO DOCKER) <<<")
            labeler_dir = workspace_dir / "chexpert-labeler"
            run_local_labeler(chexpert_input_csv, chexpert_output_csv, labeler_dir)
        else:
            print("\n>>> STEP 2: RUNNING CHEXPERT LABELER VIA DOCKER <<<")
            rel_input = chexpert_input_csv.relative_to(workspace_dir)
            rel_output = chexpert_output_csv.relative_to(workspace_dir)
            run_docker_labeler(str(rel_input), str(rel_output), workspace_dir)

    # Step: Aggregate
    if args.step in ["aggregate", "all"]:
        print("\n>>> STEP 3: AGGREGATING LABELS BACK TO SAMPLES <<<")
        if not chexpert_output_csv.exists():
            print(f"[!] File not found: {chexpert_output_csv}. Hay chay step 'label' truoc!")
            sys.exit(1)

        aggregate_labels(
            original_csv=input_csv,
            meta_json_path=meta_json,
            chexpert_output_csv=chexpert_output_csv,
            output_labeled_csv=final_output_csv,
            export_flat_reports=True,
            flat_reports_output=work_dir / "reports_labeled_flat.csv",
            verbose=True
        )
        print(f"[SUCCESS] Quy trinh hoan tat! File ket qua da duoc luu tai:\n -> {final_output_csv}")


if __name__ == "__main__":
    main()
