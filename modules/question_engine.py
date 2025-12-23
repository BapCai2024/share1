from __future__ import annotations

import json
import random
from typing import Any, Dict, Optional

import requests

from .constants import LEVEL_TO_TT27
from .matrix_engine import Task


def _safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parse."""
    try:
        return json.loads(s)
    except Exception:
        return None


def _offline_template(task: Task) -> Dict[str, Any]:
    """Fallback generator when no API key is provided.

    Produces valid structures but generic content; GV nên chỉnh sửa.
    """
    qt = task.qtype
    lv = task.level
    base = {
        "task_id": task.task_id,
        "mon": task.mon,
        "chu_de": task.chu_de,
        "bai": task.bai,
        "yccd": task.yccd,
        "qtype": qt,
        "level": lv,
        "points": task.points,
        "content": {},
    }

    if qt == "Trắc nghiệm nhiều lựa chọn":
        base["content"] = {
            "stem": f"({LEVEL_TO_TT27[lv]}) Dựa vào kiến thức ở bài '{task.bai}', chọn đáp án đúng.",
            "options": {"A": "Phương án A", "B": "Phương án B", "C": "Phương án C", "D": "Phương án D"},
            "answer": random.choice(["A", "B", "C", "D"]),
        }
    elif qt == "Đúng/Sai":
        base["content"] = {
            "stem": f"({LEVEL_TO_TT27[lv]}) Đánh dấu Đúng (Đ) hoặc Sai (S):",
            "statements": [
                {"text": "Mệnh đề 1 ...", "answer": True},
                {"text": "Mệnh đề 2 ...", "answer": False},
                {"text": "Mệnh đề 3 ...", "answer": True},
                {"text": "Mệnh đề 4 ...", "answer": False},
            ],
        }
    elif qt == "Nối cột":
        base["content"] = {
            "stem": f"({LEVEL_TO_TT27[lv]}) Nối cột A với cột B cho phù hợp:",
            "left": ["A1", "A2", "A3", "A4"],
            "right": ["B1", "B2", "B3", "B4"],
            "mapping": {"1": "1", "2": "2", "3": "3", "4": "4"},
        }
    elif qt == "Điền khuyết":
        base["content"] = {
            "stem": f"({LEVEL_TO_TT27[lv]}) Hoàn thành câu sau:",
            "text": "... ____ ...",
            "answer": "(đáp án)",
        }
    elif qt == "Tự luận":
        if str(task.mon).strip().lower() == "toán":
            prompt = "Giải bài toán có lời văn: (GV chỉnh dữ liệu/đề bài)"
        else:
            prompt = "Trình bày ngắn gọn theo yêu cầu (GV chỉnh)."
        base["content"] = {
            "prompt": f"({LEVEL_TO_TT27[lv]}) {prompt}",
            "rubric": ["Ý 1 ...", "Ý 2 ..."],
        }
    else:
        base["content"] = {"stem": "(Nội dung)"}

    return base


def _gemini_generate_content(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.6,
    max_output_tokens: int = 1200,
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
) -> str:
    """Call Gemini Developer API (AI Studio key) via REST generateContent."""
    url = api_base.rstrip("/") + f"/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return ""
    return str(parts[0].get("text") or "")


def build_generation_prompt_text(task: Task, extra_constraints: str = "") -> str:
    """Build a single prompt string for Gemini REST."""
    tt = LEVEL_TO_TT27.get(task.level, task.level)

    schema_hint = {
        "Trắc nghiệm nhiều lựa chọn": {
            "content": {"stem": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "A"}
        },
        "Đúng/Sai": {"content": {"stem": "...", "statements": [{"text": "...", "answer": True}]}} ,
        "Nối cột": {"content": {"stem": "...", "left": ["..."], "right": ["..."], "mapping": {"1": "A"}}},
        "Điền khuyết": {"content": {"stem": "...", "text": "... ____ ...", "answer": "..."}},
        "Tự luận": {"content": {"prompt": "...", "rubric": ["..."]}},
    }.get(task.qtype, {})

    sys = (
        "Bạn là chuyên gia ra đề tiểu học theo CTGDPT 2018 và Thông tư 27. "
        "Hãy tạo 1 câu hỏi đúng chuẩn sư phạm, câu văn rõ ràng, logic chặt chẽ, bám sát YCCĐ.\n"
        "YÊU CẦU BẮT BUỘC:\n"
        "- Chỉ xuất ra 1 JSON HỢP LỆ (không thêm văn bản ngoài JSON).\n"
        "- Đúng cấu trúc theo schema gợi ý.\n"
        "- Nếu là 'Nối cột': phải có đủ danh sách Cột A (left), Cột B (right) và mapping đáp án.\n"
        "- Nếu là môn Toán + dạng 'Tự luận': bắt buộc là bài toán có lời văn, dữ kiện đủ, có yêu cầu rõ.\n"
    )

    user = {
        "task_id": task.task_id,
        "mon": task.mon,
        "chu_de": task.chu_de,
        "bai": task.bai,
        "yccd": task.yccd,
        "qtype": task.qtype,
        "level": task.level,
        "level_tt27": tt,
        "points": task.points,
        "requirements": [
            "Ngôn ngữ phù hợp lớp 5, rõ ràng, không đánh đố.",
            "Bám đúng YCCĐ; không dùng kiến thức ngoài phạm vi.",
        ],
        "schema_hint": schema_hint,
        "extra_constraints": extra_constraints,
    }

    return (
        sys
        + "\n\nDỮ LIỆU NHIỆM VỤ (JSON):\n"
        + json.dumps(user, ensure_ascii=False)
        + "\n\nHãy trả về JSON hợp lệ theo schema_hint." 
    )


def generate_question(
    task: Task,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    temperature: float = 0.6,
    max_output_tokens: int = 1200,
    extra_constraints: str = "",
) -> Dict[str, Any]:
    """Generate 1 question.

    - Nếu không có api_key: dùng template offline.
    - Nếu có api_key: gọi Gemini API (AI Studio key) qua REST generateContent.
    """
    if not api_key:
        return _offline_template(task)

    prompt = build_generation_prompt_text(task, extra_constraints=extra_constraints)
    try:
        txt = _gemini_generate_content(
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_base=api_base,
        )
        obj = _safe_json_load(txt)
        if isinstance(obj, dict):
            # Ensure required metadata exists
            obj.setdefault("task_id", task.task_id)
            obj.setdefault("mon", task.mon)
            obj.setdefault("chu_de", task.chu_de)
            obj.setdefault("bai", task.bai)
            obj.setdefault("yccd", task.yccd)
            obj.setdefault("qtype", task.qtype)
            obj.setdefault("level", task.level)
            obj.setdefault("points", task.points)
            obj.setdefault("content", {})
            return obj
        return _offline_template(task)
    except Exception:
        return _offline_template(task)
