from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    from Levenshtein import ratio as levenshtein_ratio  # type: ignore
except Exception:  # pragma: no cover
    from difflib import SequenceMatcher

    def levenshtein_ratio(a: str, b: str) -> float:
        return float(SequenceMatcher(None, a, b).ratio())


@dataclass
class MatchResult:
    so_tiet: Optional[float]
    match_score: float
    matched_row: Optional[dict]


def _extract_bai_so(bai_text: str) -> Optional[int]:
    """Try to extract leading lesson number from text."""
    if bai_text is None:
        return None
    s = str(bai_text).strip()
    m = re.search(r"(?:Bài\s*)?(\d{1,3})", s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def match_so_tiet(
    ppct: pd.DataFrame,
    mon: str,
    bai_so_or_text: str,
    ten_bai: str = "",
    hoc_ki: Optional[int] = None,
    bo_sach: Optional[str] = None,
) -> MatchResult:
    """Match periods from PPCT for a given subject + lesson."""
    if ppct is None or ppct.empty:
        return MatchResult(None, 0.0, None)

    df = ppct.copy()
    df = df[df["mon"].astype(str).str.lower() == str(mon).strip().lower()]
    if hoc_ki is not None and "hoc_ki" in df.columns:
        df = df[df["hoc_ki"].fillna(-1).astype(int) == int(hoc_ki)]
    if bo_sach and "bo_sach" in df.columns:
        df = df[df["bo_sach"].astype(str).str.lower() == str(bo_sach).strip().lower()]

    bai_so = _extract_bai_so(bai_so_or_text)
    if bai_so is not None and "bai_so" in df.columns:
        exact = df[df["bai_so"].fillna(-1).astype(int) == int(bai_so)]
        if not exact.empty:
            row = exact.iloc[0].to_dict()
            so_tiet = row.get("so_tiet")
            try:
                so_tiet = float(so_tiet) if so_tiet == so_tiet else None
            except Exception:
                so_tiet = None
            return MatchResult(so_tiet, 1.0, row)

    candidate_text = (ten_bai or bai_so_or_text or "").strip().lower()
    if not candidate_text:
        return MatchResult(None, 0.0, None)

    best_sc = 0.0
    best_row = None
    for _, r in df.iterrows():
        t = str(r.get("ten_bai", "")).strip().lower()
        if not t:
            continue
        sc = levenshtein_ratio(candidate_text, t)
        if sc > best_sc:
            best_sc, best_row = sc, r

    if best_row is None:
        return MatchResult(None, 0.0, None)

    row = best_row.to_dict()
    so_tiet = row.get("so_tiet")
    try:
        so_tiet = float(so_tiet) if so_tiet == so_tiet else None
    except Exception:
        so_tiet = None
    return MatchResult(so_tiet, float(best_sc), row)
