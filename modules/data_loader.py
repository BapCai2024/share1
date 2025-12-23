from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass
class DataPaths:
    repo_root: Path

    @property
    def yccd_xlsx(self) -> Path:
        return self.repo_root / "data" / "yccd" / "khoi5_normalized.xlsx"

    @property
    def yccd_csv(self) -> Path:
        return self.repo_root / "data" / "yccd" / "khoi5_normalized.csv"

    @property
    def ppct_master(self) -> Path:
        return self.repo_root / "data" / "ppct" / "ppct_grade5_master.csv"


def load_yccd(path: Path) -> pd.DataFrame:
    """Load normalized YCCĐ.

    Expected columns (Vietnamese):
    - Môn
    - Chủ đề/Chủ điểm
    - Bài
    - Tên bài học
    - Yêu cầu cần đạt
    """
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file YCCĐ: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    required = ["Môn", "Chủ đề/Chủ điểm", "Bài", "Tên bài học", "Yêu cầu cần đạt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong YCCĐ: {missing}. Hiện có: {list(df.columns)}")

    # Normalize types
    df = df.copy()
    df["Bài"] = df["Bài"].astype(str).str.strip()
    df["Môn"] = df["Môn"].astype(str).str.strip()
    df["Chủ đề/Chủ điểm"] = df["Chủ đề/Chủ điểm"].astype(str).str.strip()
    df["Tên bài học"] = df["Tên bài học"].astype(str).str.strip()
    df["Yêu cầu cần đạt"] = df["Yêu cầu cần đạt"].astype(str).str.strip()
    return df


def load_ppct_master(path: Path) -> pd.DataFrame:
    """Load PPCT mapping (lesson -> periods)."""
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "lop",
                "hoc_ki",
                "mon",
                "bo_sach",
                "chu_de",
                "bai_so",
                "ten_bai",
                "so_tiet",
                "nguon",
            ]
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    # normalize columns
    for col in ["mon", "bo_sach", "chu_de", "ten_bai", "nguon"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "bai_so" in df.columns:
        df["bai_so"] = pd.to_numeric(df["bai_so"], errors="coerce")
    if "so_tiet" in df.columns:
        df["so_tiet"] = pd.to_numeric(df["so_tiet"], errors="coerce")
    return df


def save_ppct_master(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
