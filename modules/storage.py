from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def export_session_json(
    matrix_df: pd.DataFrame,
    blueprint: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> str:
    payload = {
        "version": "v1.1",
        "meta": meta,
        "matrix": matrix_df.fillna("" ).to_dict(orient="records"),
        "blueprint": blueprint,
        "questions": questions,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_session_json(text: str) -> Dict[str, Any]:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Session JSON không hợp lệ")
    return obj


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
