from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import LEVELS, QTYPE_ORDER


def qcol(qtype: str, level: str) -> str:
    return f"{qtype}|{level}"


MATRIX_BASE_COLS = [
    "TT",
    "Môn",
    "Chủ đề",
    "Bài/Nội dung",
    "YCCĐ",
    "Số tiết",
    "Block",
    "Tỉ lệ (%)",
    "Số điểm cần đạt",
    "Tổng số câu/ý",
    "Điểm từng bài",
]


def all_qcols() -> List[str]:
    cols = []
    for qt in QTYPE_ORDER:
        for lv in LEVELS:
            cols.append(qcol(qt, lv))
    return cols


def init_empty_matrix() -> pd.DataFrame:
    cols = MATRIX_BASE_COLS + all_qcols()
    df = pd.DataFrame(columns=cols)
    return df


def normalize_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all columns exist and correct dtypes."""
    if df is None or df.empty:
        return init_empty_matrix()

    df = df.copy()
    for c in MATRIX_BASE_COLS:
        if c not in df.columns:
            df[c] = None
    for c in all_qcols():
        if c not in df.columns:
            df[c] = 0

    # numeric columns
    for c in ["Số tiết", "Tỉ lệ (%)", "Số điểm cần đạt", "Tổng số câu/ý", "Điểm từng bài", "Block"] + all_qcols():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # text columns
    for c in ["Môn", "Chủ đề", "Bài/Nội dung", "YCCĐ"]:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({"nan": ""}).str.strip()

    # TT
    if "TT" in df.columns:
        # Keep existing TT if present, else fill later
        pass

    return df


def recompute_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    qcols = all_qcols()
    df["Tổng số câu/ý"] = df[qcols].fillna(0).sum(axis=1)
    return df


def compute_ratio_and_points(
    df: pd.DataFrame,
    mode: str = "whole_exam_10",
    total_points: float = 10.0,
    block_points: Dict[int, float] | None = None,
) -> pd.DataFrame:
    """Compute 'Tỉ lệ (%)' and 'Số điểm cần đạt'.

    mode:
      - whole_exam_10: ratio over all selected rows; points = ratio * total_points / 100
      - blocks: ratio within each Block; points = ratio * block_points[block]/100
    """
    df = df.copy()
    if "Số tiết" not in df.columns:
        return df

    df["Số tiết"] = pd.to_numeric(df["Số tiết"], errors="coerce")

    if mode == "blocks":
        if block_points is None:
            block_points = {1: total_points}
        if "Block" not in df.columns:
            df["Block"] = 1
        df["Block"] = pd.to_numeric(df["Block"], errors="coerce").fillna(1).astype(int)

        ratios = []
        pts = []
        for idx, row in df.iterrows():
            b = int(row.get("Block", 1))
            denom = df[df["Block"] == b]["Số tiết"].sum(skipna=True)
            if denom and denom > 0 and row.get("Số tiết") == row.get("Số tiết"):
                r = float(row["Số tiết"]) / float(denom) * 100.0
            else:
                r = np.nan
            ratios.append(r)
            bp = float(block_points.get(b, total_points))
            pts.append(r * bp / 100.0 if r == r else np.nan)

        df["Tỉ lệ (%)"] = ratios
        df["Số điểm cần đạt"] = pts

    else:
        denom = df["Số tiết"].sum(skipna=True)
        if denom and denom > 0:
            df["Tỉ lệ (%)"] = df["Số tiết"] / float(denom) * 100.0
            df["Số điểm cần đạt"] = df["Tỉ lệ (%)"] * float(total_points) / 100.0
        else:
            df["Tỉ lệ (%)"] = np.nan
            df["Số điểm cần đạt"] = np.nan

    # default: Điểm từng bài = Số điểm cần đạt (có thể override)
    df["Điểm từng bài"] = df["Số điểm cần đạt"]
    return df


@dataclass
class Task:
    task_id: str
    mon: str
    chu_de: str
    bai: str
    yccd: str
    qtype: str
    level: str
    points: float


def build_blueprint_from_matrix(df: pd.DataFrame, default_points_by_qtype: Dict[str, float]) -> List[Task]:
    """Create a list of generation tasks based on distribution columns in the matrix."""
    df = normalize_matrix(df)
    df = recompute_totals(df)
    tasks: List[Task] = []
    counter = 1

    for _, row in df.iterrows():
        mon = str(row.get("Môn", "")).strip()
        chu_de = str(row.get("Chủ đề", "")).strip()
        bai = str(row.get("Bài/Nội dung", "")).strip()
        yccd = str(row.get("YCCĐ", "")).strip()

        for qt in QTYPE_ORDER:
            for lv in LEVELS:
                c = qcol(qt, lv)
                n = row.get(c, 0)
                try:
                    n = int(float(n)) if n == n else 0
                except Exception:
                    n = 0
                if n <= 0:
                    continue

                pt = float(default_points_by_qtype.get(qt, 1.0))
                for _k in range(n):
                    tid = f"T{counter:04d}"
                    tasks.append(Task(tid, mon, chu_de, bai, yccd, qt, lv, pt))
                    counter += 1

    return tasks
