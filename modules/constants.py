"""Constants and shared enums.

V1.1 aligns levels with TT27:
- Biết    ~ Mức 1 (Nhận biết)
- Hiểu    ~ Mức 2 (Kết nối/Sắp xếp)
- VD      ~ Mức 3 (Vận dụng)

These labels are kept as 'Biết/Hiểu/VD' because teachers usually build matrices with them.
"""

GRADE_DEFAULT = 5

LEVELS = ["Biết", "Hiểu", "VD"]
LEVEL_TO_TT27 = {
    "Biết": "Mức 1 (Nhận biết)",
    "Hiểu": "Mức 2 (Kết nối/Sắp xếp)",
    "VD": "Mức 3 (Vận dụng)",
}

QTYPE_ORDER = [
    "Trắc nghiệm nhiều lựa chọn",
    "Đúng/Sai",
    "Nối cột",
    "Điền khuyết",
    "Tự luận",
]

DEFAULT_STEP = 0.5

# ND30 basics (can be adjusted)
ND30_MARGINS_CM = {"top": 2.0, "bottom": 2.0, "left": 3.0, "right": 2.0}
ND30_FONT_NAME = "Times New Roman"
ND30_FONT_SIZE = 13
ND30_LINE_SPACING = 1.15
