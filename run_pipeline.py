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
import concurrent.futures
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from prepare_chexpert_input import prepare_input
from aggregate_chexpert_labels import aggregate_labels


def read_input_reports(input_csv: Path) -> list:
    """Read all reports from single-column CheXpert input CSV."""
    reports = []
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                reports.append(row[0])
    return reports


def write_input_reports(output_csv: Path, reports: list):
    """Write reports to headerless single-column CSV."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for text in reports:
            writer.writerow([text])


def merge_labeled_chunks(chunk_outputs: list, final_output: Path, expected_rows: int):
    """Merge labeled CSV chunks and verify row count."""
    dfs = []
    for chunk_file in chunk_outputs:
        if not chunk_file.exists():
            raise FileNotFoundError(f"Missing chunk output: {chunk_file}")
        chunk_df = pd.read_csv(chunk_file)
        dfs.append(chunk_df)
    merged_df = pd.concat(dfs, ignore_index=True)
    if len(merged_df) != expected_rows:
        raise ValueError(f"Merged rows ({len(merged_df)}) does not match expected ({expected_rows})")
    merged_df.to_csv(final_output, index=False)


def run_single_local_worker(chunk_in: Path, chunk_out: Path, labeler_dir: Path, worker_id: int, env: dict):
    """Run a single local worker for one chunk."""
    cmd = [
        sys.executable,
        str(labeler_dir / "label.py"),
        "--reports_path", str(chunk_in.resolve()),
        "--output_path", str(chunk_out.resolve()),
    ]
    res = subprocess.run(cmd, env=env, cwd=labeler_dir, capture_output=True, text=True)
    return res.returncode, res.stderr


def run_single_docker_worker(chunk_in_rel: str, chunk_out_rel: str, workspace_dir: Path, worker_id: int):
    """Run a single Docker worker for one chunk."""
    input_container = f"/data/{Path(chunk_in_rel).as_posix()}"
    output_container = f"/data/{Path(chunk_out_rel).as_posix()}"
    workspace_str = str(workspace_dir.resolve()).replace("\\", "/")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace_str}:/data",
        "chexpert-labeler:latest",
        "python", "label.py",
        "--reports_path", input_container,
        "--output_path", output_container,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stderr


def run_parallel_labeling(
    input_csv: Path,
    output_csv: Path,
    workspace_dir: Path,
    work_dir: Path,
    workers: int,
    no_docker: bool
):
    """Split input into chunks, run workers in parallel, and merge results."""
    reports = read_input_reports(input_csv)
    total_reports = len(reports)
    if total_reports == 0:
        print("[!] Input reports file is empty!")
        return

    actual_workers = min(workers, total_reports)
    print(f"\n[*] Starting PARALLEL CheXpert labeling with {actual_workers} workers...")
    print(f"[*] Total unique reports to label: {total_reports}")

    chunks_dir = work_dir / "_chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = (total_reports + actual_workers - 1) // actual_workers
    chunk_tasks = []
    chunk_output_paths = []

    for i in range(actual_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_reports)
        if start_idx >= total_reports:
            break
        chunk_reports = reports[start_idx:end_idx]
        chunk_in = chunks_dir / f"chunk_{i:04d}_in.csv"
        chunk_out = chunks_dir / f"chunk_{i:04d}_out.csv"
        write_input_reports(chunk_in, chunk_reports)
        chunk_tasks.append((i, chunk_in, chunk_out, len(chunk_reports)))
        chunk_output_paths.append(chunk_out)

    num_active_chunks = len(chunk_tasks)
    print(f"[*] Partitioned dataset into {num_active_chunks} chunks (~{chunk_size} reports/chunk).\n")

    start_time = time.time()

    if no_docker:
        labeler_dir = workspace_dir / "chexpert-labeler"
        env = os.environ.copy()
        negbio_dir = labeler_dir / "NegBio"
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{negbio_dir}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(negbio_dir)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_active_chunks) as executor:
            futures = {
                executor.submit(run_single_local_worker, chunk_in, chunk_out, labeler_dir, worker_id, env): (worker_id, count)
                for worker_id, chunk_in, chunk_out, count in chunk_tasks
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                worker_id, count = futures[future]
                ret, stderr = future.result()
                if ret != 0:
                    err_msg = stderr.strip() if stderr else "No stderr output."
                    raise RuntimeError(f"Local worker {worker_id} failed with exit code {ret}!\n\n--- WORKER ERROR LOG ---\n{err_msg}\n-------------------------\n[!] Gợi ý: Nếu đang chạy trên Windows, hãy dùng Docker (bỏ cờ --no_docker). Cờ --no_docker chỉ dùng trên Linux Server có cài Python 3.7.")
                completed += 1
                elapsed = time.time() - start_time
                print(f" -> [Progress {completed}/{num_active_chunks}] Worker {worker_id+1} finished ({count} reports) | Elapsed: {elapsed:.1f}s")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_active_chunks) as executor:
            futures = {
                executor.submit(
                    run_single_docker_worker,
                    str(chunk_in.relative_to(workspace_dir)),
                    str(chunk_out.relative_to(workspace_dir)),
                    workspace_dir,
                    worker_id
                ): (worker_id, count)
                for worker_id, chunk_in, chunk_out, count in chunk_tasks
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                worker_id, count = futures[future]
                ret, stderr = future.result()
                if ret != 0:
                    err_msg = stderr.strip() if stderr else "No stderr output."
                    raise RuntimeError(f"Docker worker {worker_id} failed with exit code {ret}!\n\n--- DOCKER ERROR LOG ---\n{err_msg}\n-------------------------\n[!] Hãy đảm bảo Docker Desktop đang chạy và image 'chexpert-labeler:latest' đã được build.")
                completed += 1
                elapsed = time.time() - start_time
                print(f" -> [Progress {completed}/{num_active_chunks}] Docker Worker {worker_id+1} finished ({count} reports) | Elapsed: {elapsed:.1f}s")

    total_elapsed = time.time() - start_time
    rate = total_reports / total_elapsed if total_elapsed > 0 else 0
    print(f"\n[*] All {num_active_chunks} workers finished in {total_elapsed:.1f}s (~{rate:.1f} reports/sec)!")
    print(f"[*] Merging chunk outputs -> {output_csv} ...")
    merge_labeled_chunks(chunk_output_paths, output_csv, total_reports)

    shutil.rmtree(chunks_dir, ignore_errors=True)
    print(f"[OK] Parallel labeling completed successfully -> {output_csv}\n")


def run_docker_labeler(
    input_csv_rel: str,
    output_csv_rel: str,
    workspace_dir: Path
):
    """Run chexpert-labeler inside Docker container (single worker)."""
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
    """Run chexpert-labeler directly on the host using local Python (single worker)."""
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

    res = subprocess.run(cmd, env=env, cwd=labeler_dir)
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
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes for labeling (default: 1). "
                             "Set to e.g. 8, 16, 24 (or 0 for auto all CPU cores) for 10x-20x speedup.")
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

    # Step: Label (Docker or Local Python, Single or Parallel)
    if args.step in ["label", "all"]:
        if not chexpert_input_csv.exists():
            print(f"[!] File not found: {chexpert_input_csv}. Hay chay step 'prepare' truoc!")
            sys.exit(1)

        actual_workers = args.workers
        if actual_workers <= 0:
            actual_workers = max(1, os.cpu_count() or 1)

        if actual_workers > 1:
            mode_str = "LOCAL PYTHON" if args.no_docker else "DOCKER"
            print(f"\n>>> STEP 2: RUNNING PARALLEL CHEXPERT LABELER ({actual_workers} WORKERS, {mode_str}) <<<")
            run_parallel_labeling(
                input_csv=chexpert_input_csv,
                output_csv=chexpert_output_csv,
                workspace_dir=workspace_dir,
                work_dir=work_dir,
                workers=actual_workers,
                no_docker=args.no_docker
            )
        else:
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
