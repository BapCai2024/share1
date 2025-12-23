"""Extract a *rough* PPCT table from K5.pdf.

PDF tables may be messy. This script tries to find patterns like:
- "Bài 12 ... (3 tiết)"
- "Bài 12. ... (Tiết 1)" ...

Output: data/ppct/ppct_from_k5_pdf.csv

Usage:
  python scripts/build_ppct_from_k5_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = REPO_ROOT / "data" / "ppct" / "sources" / "K5.pdf"
OUT_PATH = REPO_ROOT / "data" / "ppct" / "ppct_from_k5_pdf.csv"

SUBJECT_HINTS = [
    ("Tiếng Việt", ["Tiếng Việt", "Tập đọc", "Luyện từ", "Chính tả", "Tập làm văn"]),
    ("Toán", ["Toán", "hình học", "đo lường", "số thập phân", "phân số"]),
    ("Khoa học", ["Khoa học", "Vật", "Chất", "cơ thể", "môi trường"]),
    ("Lịch sử và Địa lí", ["Lịch sử", "Địa lí", "bản đồ", "vùng"]),
    ("Tin học", ["Tin học", "máy tính", "phần mềm", "tệp"]),
    ("Công nghệ", ["Công nghệ", "thiết bị", "an toàn", "điện"]),
]


def guess_subject(text: str) -> str:
    t = (text or "").lower()
    scores = []
    for name, keys in SUBJECT_HINTS:
        sc = sum(1 for k in keys if k.lower() in t)
        scores.append((sc, name))
    scores.sort(reverse=True)
    return scores[0][1] if scores and scores[0][0] > 0 else ""


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy: {PDF_PATH}")

    reader = PdfReader(str(PDF_PATH))
    rows = []

    # Pattern 1: Bài 12 (3 tiết)
    p1 = re.compile(r"Bài\s*(\d{1,3})\s*[^\n\r]*?\((\d+)\s*tiết\)", re.IGNORECASE)

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        subj = guess_subject(text)
        for m in p1.finditer(text):
            bai_so = int(m.group(1))
            so_tiet = int(m.group(2))
            snippet = m.group(0)
            rows.append(
                {
                    "lop": 5,
                    "hoc_ki": "",
                    "mon": subj,
                    "bo_sach": "",
                    "chu_de": "",
                    "bai_so": bai_so,
                    "ten_bai": "",
                    "so_tiet": so_tiet,
                    "page": i + 1,
                    "snippet": snippet,
                    "nguon": "K5.pdf (auto)"
                }
            )

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
