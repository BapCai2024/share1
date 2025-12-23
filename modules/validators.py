from __future__ import annotations

from typing import Dict, List, Tuple

from .constants import LEVELS, QTYPE_ORDER


def validate_question(q: Dict) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings)."""
    errors: List[str] = []
    warnings: List[str] = []

    qtype = str(q.get("qtype", "")).strip()
    level = str(q.get("level", "")).strip()
    content = q.get("content", {}) or {}

    if qtype not in QTYPE_ORDER:
        errors.append(f"qtype không hợp lệ: {qtype}")
    if level not in LEVELS:
        errors.append(f"level không hợp lệ: {level}")

    # Basic
    stem = str(content.get("stem", "")).strip()
    if qtype != "Tự luận" and not stem:
        errors.append("Thiếu nội dung câu hỏi (stem)")

    if qtype == "Trắc nghiệm nhiều lựa chọn":
        options = content.get("options", {}) or {}
        if not isinstance(options, dict) or len(options) < 4:
            errors.append("MCQ phải có options A-D")
        ans = str(content.get("answer", "")).strip().upper()
        if ans not in ["A", "B", "C", "D"]:
            errors.append("MCQ phải có đáp án A/B/C/D")
        # Unique-ish
        opt_texts = [str(options.get(k, "")).strip() for k in ["A", "B", "C", "D"]]
        if len({t for t in opt_texts if t}) < 4:
            warnings.append("MCQ có lựa chọn bị trùng/thiếu")

    elif qtype == "Đúng/Sai":
        statements = content.get("statements", []) or []
        if not isinstance(statements, list) or len(statements) < 2:
            errors.append("Đúng/Sai phải có ít nhất 2 mệnh đề")
        else:
            for i, st in enumerate(statements, start=1):
                txt = str(st.get("text", "")).strip()
                if not txt:
                    errors.append(f"Mệnh đề {i} bị trống")
                if st.get("answer") not in [True, False, "Đ", "S", "Đúng", "Sai", "True", "False"]:
                    errors.append(f"Mệnh đề {i} thiếu đáp án Đ/S")

    elif qtype == "Nối cột":
        left = content.get("left", []) or []
        right = content.get("right", []) or []
        mapping = content.get("mapping", {}) or {}
        if not isinstance(left, list) or len(left) < 2:
            errors.append("Nối cột phải có cột A (left) >= 2 mục")
        if not isinstance(right, list) or len(right) < 2:
            errors.append("Nối cột phải có cột B (right) >= 2 mục")
        if not isinstance(mapping, dict) or len(mapping) < min(len(left), len(right)):
            warnings.append("Nối cột: mapping đáp án có thể chưa đủ")

    elif qtype == "Điền khuyết":
        text = str(content.get("text", "")).strip()
        answer = content.get("answer")
        if "____" not in text and "…" not in text:
            warnings.append("Điền khuyết nên có ký hiệu chỗ trống (____ hoặc …)")
        if answer is None or str(answer).strip() == "":
            errors.append("Điền khuyết phải có đáp án")

    elif qtype == "Tự luận":
        prompt = str(content.get("prompt", "")).strip()
        rubric = content.get("rubric", []) or []
        if not prompt:
            errors.append("Tự luận phải có prompt")
        if not isinstance(rubric, list) or len(rubric) == 0:
            warnings.append("Tự luận nên có rubric (ý chấm/thang điểm)")

    # Heuristic warnings for levels (optional)
    if level == "VD":
        t = (stem + " " + str(content.get("prompt", ""))).lower()
        # very light heuristic: encourage a context
        if len(t) < 30:
            warnings.append("Mức VD thường cần bối cảnh/tình huống rõ hơn")

    return errors, warnings


def validate_exam_points(questions: List[Dict]) -> Tuple[List[str], List[str]]:
    """Validate total points (TT27: thang 10, không điểm thập phân).

    V1.1: cho phép 0.5 nội bộ, nhưng khuyến nghị tổng điểm ra số nguyên.
    """
    errs: List[str] = []
    warns: List[str] = []

    total = 0.0
    for q in questions:
        try:
            total += float(q.get("points", 0))
        except Exception:
            pass

    # floating tolerance
    if abs(total - 10.0) > 1e-6:
        warns.append(f"Tổng điểm hiện tại = {total:g} (khuyến nghị = 10)")

    if abs(total - round(total)) > 1e-6:
        warns.append("Tổng điểm không phải số nguyên (TT27 khuyến nghị không dùng điểm thập phân)")

    return errs, warns
