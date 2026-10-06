"""
Robust CSV reader for MIMIC-CXR augmented datasets (mimic_cxr_aug_train.csv, etc.)

Xử lý triệt để lỗi tokenizing khi các cột Python list không được quote bằng dấu ngoặc kép.
"""

from pathlib import Path
from typing import Optional, Union, List
import ast
import csv
import re
import pandas as pd


import json


def _safe_parse_list(value: str) -> list:
    """Parse string dạng Python list literal "['a', 'b']" hoặc JSON string sang list thực tế."""
    if pd.isna(value) or not isinstance(value, str):
        return []
    value = value.strip()
    if not value or value == "[]":
        return []
    # Thử json.loads trước (cho trường hợp JSON string như report_labels có chứa null)
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        pass
    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else [result]
    except (ValueError, SyntaxError):
        # Fallback: trích xuất các phần tử giữa dấu nháy đơn
        return re.findall(r"'([^']*)'", value)


def _parse_line_with_brackets(line: str, expected_cols: int) -> Optional[List[str]]:
    """
    Parse một dòng CSV có chứa các cột Python list [...] không được quote chuẩn.
    Dùng bracket-tracking để phân tách chính xác các cột.
    """
    first_bracket = line.find('[')
    if first_bracket == -1:
        return None

    # Các cột trước dấu '[' đầu tiên (thường là các cột index, subject_id)
    prefix = line[:first_bracket].rstrip(',')
    prefix_cols = prefix.split(',')
    rest = line[first_bracket:].rstrip('\r\n')

    columns = []
    current = []
    depth = 0

    for char in rest:
        if char == '[':
            depth += 1
            current.append(char)
        elif char == ']':
            depth -= 1
            current.append(char)
            if depth == 0:
                columns.append(''.join(current))
                current = []
        elif char == ',' and depth == 0:
            if not current:
                continue
            columns.append(''.join(current))
            current = []
        else:
            current.append(char)

    if current:
        remaining = ''.join(current).strip()
        if remaining:
            columns.append(remaining)

    total_cols = prefix_cols + columns
    if len(total_cols) != expected_cols:
        return None

    return total_cols


def load_csv(
    filepath: Union[str, Path],
    parse_lists: bool = True,
    drop_unnamed: bool = True,
    nrows: Optional[int] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Đọc file CSV một cách robust, hỗ trợ các file bị lỗi quoting ở các cột list.

    Args:
        filepath (str | Path): Đường dẫn đến file CSV cần đọc.
        parse_lists (bool): Có chuyển đổi các cột list (dạng string) sang Python list không. Mặc định là True.
        drop_unnamed (bool): Tự động loại bỏ các cột index thừa ('Unnamed: 0', 'Unnamed: 0.1'). Mặc định là True.
        nrows (int, optional): Giới hạn số dòng cần đọc (hữu ích khi test nhanh). Mặc định là None (đọc hết).
        verbose (bool): In thông tin trong quá trình đọc file. Mặc định là False.

    Returns:
        pd.DataFrame: DataFrame đã được làm sạch và parse dữ liệu chuẩn.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    rows = []
    skipped = 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        header_line = f.readline().strip()
        header = header_line.split(",")
        num_cols = len(header)

        for line_num, raw_line in enumerate(f, start=2):
            if nrows is not None and len(rows) >= nrows:
                break

            if not raw_line.strip():
                continue

            # Nếu dòng có chứa dấu ngoặc kép -> ưu tiên dùng csv.reader chuẩn
            if '"' in raw_line:
                try:
                    parsed = list(csv.reader([raw_line]))[0]
                    if len(parsed) == num_cols:
                        rows.append(parsed)
                        continue
                except Exception:
                    pass

            # Dòng không có ngoặc kép hoặc csv.reader thất bại -> fallback bracket-tracking
            parsed = _parse_line_with_brackets(raw_line, num_cols)
            if parsed and len(parsed) == num_cols:
                rows.append(parsed)
            else:
                skipped += 1
                if verbose and skipped <= 5:
                    print(f"[!] Bỏ qua dòng {line_num}: không parse được")

    df = pd.DataFrame(rows, columns=header)

    # Chuyển đổi kiểu dữ liệu số nếu có
    if "subject_id" in df.columns:
        df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")

    for col in ["Unnamed: 0.1", "Unnamed: 0"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse các cột chứa Python list literal
    if parse_lists:
        list_columns = ["image", "view", "AP", "PA", "Lateral", "text", "text_augment", "report_labels"]
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(_safe_parse_list)

    # Bỏ cột index thừa nếu cần
    if drop_unnamed:
        cols_to_drop = [c for c in ["Unnamed: 0.1", "Unnamed: 0"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    if verbose:
        print(f"[OK] Da doc thanh cong {len(df)} dong, {len(df.columns)} cot tu: {path.name}")
        if skipped > 0:
            print(f"[!] Da bo qua {skipped} dong loi.")

    return df


# Alias tương thích ngược
read_csv = load_csv


if __name__ == "__main__":
    default_path = Path(__file__).parent / "mimic_cxr_aug_train.csv"
    if default_path.exists():
        print(f"Đang đọc thử file: {default_path.name} ...")
        df = load_csv(default_path, verbose=True)
        print(f"Hoàn thành! Shape: {df.shape}")
        print("\nThông tin 5 cột đầu tiên:")
        print(df.head(1).T)
