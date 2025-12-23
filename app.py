from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from modules.constants import GRADE_DEFAULT, LEVELS, QTYPE_ORDER
from modules.data_loader import DataPaths, load_ppct_master, load_yccd, save_ppct_master
from modules.excel_export import export_matrix_to_excel
from modules.matching import match_so_tiet
from modules.matrix_engine import (
    all_qcols,
    build_blueprint_from_matrix,
    compute_ratio_and_points,
    init_empty_matrix,
    normalize_matrix,
    recompute_totals,
)
from modules.question_engine import generate_question
from modules.storage import export_session_json, import_session_json
from modules.validators import validate_exam_points, validate_question
from modules.word_export import build_blueprint_docx, build_exam_docx


REPO_ROOT = Path(__file__).resolve().parent
PATHS = DataPaths(REPO_ROOT)


def _init_state() -> None:
    if "yccd" not in st.session_state:
        st.session_state.yccd = None
    if "ppct" not in st.session_state:
        st.session_state.ppct = None
    if "matrix" not in st.session_state:
        st.session_state.matrix = init_empty_matrix()
    if "matrix_locked" not in st.session_state:
        st.session_state.matrix_locked = False
    if "blueprint" not in st.session_state:
        st.session_state.blueprint = []  # list[dict]
    if "questions" not in st.session_state:
        st.session_state.questions = []  # list[dict]
    if "tpl_exam" not in st.session_state:
        st.session_state.tpl_exam = None
    if "tpl_blueprint" not in st.session_state:
        st.session_state.tpl_blueprint = None
    if "meta" not in st.session_state:
        st.session_state.meta = {
            "title": "ĐỀ KIỂM TRA ĐỊNH KÌ",
            "blueprint_title": "BẢNG ĐẶC TẢ ĐỀ",
            "grade": GRADE_DEFAULT,
            "subject": "",
            "duration": "",
        }


def _load_default_data() -> None:
    if st.session_state.yccd is None:
        st.session_state.yccd = load_yccd(PATHS.yccd_xlsx if PATHS.yccd_xlsx.exists() else PATHS.yccd_csv)
    if st.session_state.ppct is None:
        st.session_state.ppct = load_ppct_master(PATHS.ppct_master)


def _sidebar() -> Dict[str, Any]:
    st.sidebar.header("Cấu hình")

    # LLM API (Gemini)
    api_key_input = st.sidebar.text_input("Gemini API key (tuỳ chọn)", type="password")
    api_key = api_key_input or st.secrets.get("GEMINI_API_KEY", "")
    model = st.sidebar.text_input("Model", value=st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash"))
    api_base = st.sidebar.text_input(
        "API base",
        value=st.secrets.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
        help="Mặc định dùng Gemini Developer API (AI Studio key).",
    )
    temperature = st.sidebar.slider("Độ sáng tạo", 0.0, 1.0, 0.6, 0.05)
    max_output_tokens = st.sidebar.slider("Max output tokens", 256, 4096, 1200, 128)
    if not api_key:
        st.sidebar.info("Không có API key: app sẽ sinh câu hỏi theo template offline để test pipeline.")

    st.sidebar.divider()

    # Templates
    st.sidebar.subheader("Template Word (tuỳ chọn)")
    tpl_exam = st.sidebar.file_uploader("Upload template 'Đề' (.docx)", type=["docx"], key="tpl_exam_uploader")
    tpl_blue = st.sidebar.file_uploader("Upload template 'Bảng đặc tả' (.docx)", type=["docx"], key="tpl_blue_uploader")
    if tpl_exam is not None:
        st.session_state.tpl_exam = tpl_exam.read()
        st.sidebar.success("Đã nạp template Đề")
    if tpl_blue is not None:
        st.session_state.tpl_blueprint = tpl_blue.read()
        st.sidebar.success("Đã nạp template Bảng đặc tả")

    st.sidebar.divider()

    st.sidebar.subheader("PPCT/Số tiết")
    ppct_upload = st.sidebar.file_uploader("Upload PPCT master (.csv)", type=["csv"], key="ppct_uploader")
    if ppct_upload is not None:
        df = pd.read_csv(ppct_upload, encoding="utf-8-sig")
        st.session_state.ppct = df
        save_ppct_master(df, PATHS.ppct_master)
        st.sidebar.success("Đã cập nhật PPCT master")

    st.sidebar.divider()

    st.sidebar.subheader("Import/Export session")
    sess_up = st.sidebar.file_uploader("Import session (.json)", type=["json"], key="sess_uploader")
    if sess_up is not None:
        obj = import_session_json(sess_up.read().decode("utf-8"))
        st.session_state.meta = obj.get("meta", st.session_state.meta)
        st.session_state.matrix = pd.DataFrame(obj.get("matrix", []))
        st.session_state.blueprint = obj.get("blueprint", [])
        st.session_state.questions = obj.get("questions", [])
        st.sidebar.success("Đã nạp session")

    st.sidebar.divider()

    st.sidebar.subheader("Preset nhanh (test)")
    if st.sidebar.button("Nạp ma trận mẫu"):
        preset = (PATHS.repo_root / "data" / "matrices" / "presets" / "sample_matrix.json")
        if preset.exists():
            recs = json.loads(preset.read_text(encoding="utf-8"))
            st.session_state.matrix = pd.DataFrame(recs)
            st.session_state.matrix_locked = False
            st.session_state.blueprint = []
            st.session_state.questions = []
            st.sidebar.success("Đã nạp ma trận mẫu")
        else:
            st.sidebar.error("Không tìm thấy preset")

    return {
        "api_key": api_key,
        "model": model,
        "api_base": api_base,
        "temperature": temperature,
        "max_output_tokens": int(max_output_tokens),
    }


def _meta_form():
    st.subheader("Thông tin đề")
    meta = st.session_state.meta
    c1, c2, c3 = st.columns([2, 1, 1])
    meta["title"] = c1.text_input("Tiêu đề đề", value=meta.get("title", ""))
    meta["grade"] = c2.number_input("Lớp", min_value=1, max_value=12, value=int(meta.get("grade", GRADE_DEFAULT)))
    meta["duration"] = c3.text_input("Thời gian", value=meta.get("duration", ""), placeholder="VD: 40 phút")
    meta["subject"] = st.text_input("Môn", value=meta.get("subject", ""))
    meta["blueprint_title"] = st.text_input("Tiêu đề bảng đặc tả", value=meta.get("blueprint_title", "BẢNG ĐẶC TẢ ĐỀ"))
    st.session_state.meta = meta


def _tab_matrix():
    st.header("Tab 1 – Tạo ma trận")

    yccd: pd.DataFrame = st.session_state.yccd
    ppct: pd.DataFrame = st.session_state.ppct

    _meta_form()

    st.markdown("### Thêm dòng từ kho YCCĐ")
    mcol1, mcol2, mcol3 = st.columns([1, 2, 2])
    mon = mcol1.selectbox("Môn", sorted(yccd["Môn"].unique()))
    df_m = yccd[yccd["Môn"] == mon]
    chu_de = mcol2.selectbox("Chủ đề/Chủ điểm", sorted(df_m["Chủ đề/Chủ điểm"].unique()))
    df_cd = df_m[df_m["Chủ đề/Chủ điểm"] == chu_de]

    # Lesson label
    df_cd = df_cd.copy()
    df_cd["_bai_label"] = df_cd.apply(lambda r: f"Bài {r['Bài']}: {r['Tên bài học']}", axis=1)
    bai_label = mcol3.selectbox("Bài", sorted(df_cd["_bai_label"].unique()))
    df_b = df_cd[df_cd["_bai_label"] == bai_label]
    yccd_text = st.selectbox("YCCĐ", df_b["Yêu cầu cần đạt"].unique())

    # Auto so_tiet
    hoc_ki = st.number_input("Học kì", min_value=1, max_value=2, value=1)
    bo_sach = st.text_input("Bộ sách (tuỳ chọn)", value="Kết nối tri thức")
    if st.button("➕ Thêm dòng vào ma trận", disabled=st.session_state.matrix_locked):
        matrix = normalize_matrix(st.session_state.matrix)
        # Determine so_tiet
        bai_so = df_b.iloc[0]["Bài"]
        ten_bai = df_b.iloc[0]["Tên bài học"]
        mr = match_so_tiet(ppct, mon=mon, bai_so_or_text=str(bai_so), ten_bai=str(ten_bai), hoc_ki=hoc_ki, bo_sach=bo_sach)
        so_tiet = mr.so_tiet

        record = {c: 0 for c in all_qcols()}
        record.update(
            {
                "TT": len(matrix) + 1,
                "Môn": mon,
                "Chủ đề": chu_de,
                "Bài/Nội dung": bai_label,
                "YCCĐ": yccd_text,
                "Số tiết": so_tiet if so_tiet is not None else "",
                "Block": 1,
            }
        )
        matrix = pd.concat([matrix, pd.DataFrame([record])], ignore_index=True)
        matrix = recompute_totals(matrix)
        st.session_state.matrix = matrix
        if so_tiet is None:
            st.warning("Không dò được Số tiết từ PPCT (bạn có thể nhập tay).")
        else:
            st.success(f"Đã thêm dòng. Auto Số tiết = {so_tiet} (match score {mr.match_score:.2f})")

    st.markdown("### Bảng ma trận")

    matrix = normalize_matrix(st.session_state.matrix)

    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    mode = c1.selectbox("Chế độ tính", ["whole_exam_10", "blocks"], format_func=lambda x: "Toàn đề (10đ)" if x == "whole_exam_10" else "Theo Block")
    total_points = c2.number_input("Tổng điểm", min_value=1.0, max_value=10.0, value=10.0, step=0.5)
    block1 = c3.number_input("Điểm Block 1", min_value=0.0, max_value=10.0, value=10.0, step=0.5)
    block2 = c4.number_input("Điểm Block 2", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

    if st.button("🧮 Tính tỉ lệ – số điểm", disabled=st.session_state.matrix_locked):
        bp = {1: float(block1)}
        if block2 > 0:
            bp[2] = float(block2)
        matrix = compute_ratio_and_points(matrix, mode=mode, total_points=float(total_points), block_points=bp)
        matrix = recompute_totals(matrix)
        st.session_state.matrix = matrix
        st.success("Đã cập nhật Tỉ lệ và Số điểm")

    lock_col1, lock_col2 = st.columns([1, 3])
    if lock_col1.button("🔒 Chốt ma trận" if not st.session_state.matrix_locked else "🔓 Mở khoá"):
        st.session_state.matrix_locked = not st.session_state.matrix_locked

    # Data editor
    edited = st.data_editor(
        matrix,
        use_container_width=True,
        disabled=st.session_state.matrix_locked,
        num_rows="dynamic",
    )
    st.session_state.matrix = recompute_totals(edited)


def _tab_questions(cfg: Dict[str, Any]):
    st.header("Tab 2 – Tạo câu hỏi")

    matrix = normalize_matrix(st.session_state.matrix)
    if matrix.empty:
        st.info("Chưa có dữ liệu ma trận. Hãy thêm dòng ở Tab 1.")
        return

    st.markdown("### Phân bố số câu theo dạng × mức")
    st.caption("Nhập số lượng câu ở các cột dạng|mức trong bảng ma trận (Tab 1).")

    # Default point per qtype
    st.markdown("### Điểm mặc định theo dạng câu hỏi")
    pts_cols = st.columns(len(QTYPE_ORDER))
    default_points = {}
    for i, qt in enumerate(QTYPE_ORDER):
        default_points[qt] = pts_cols[i].number_input(qt, min_value=0.5, max_value=10.0, value=0.5 if qt == "Trắc nghiệm nhiều lựa chọn" else 1.0, step=0.5)

    st.divider()

    # Build blueprint
    if st.button("🧩 Tạo blueprint từ ma trận"):
        tasks = build_blueprint_from_matrix(matrix, default_points_by_qtype=default_points)
        st.session_state.blueprint = [t.__dict__ for t in tasks]
        st.success(f"Blueprint: {len(tasks)} task")

    if not st.session_state.blueprint:
        st.info("Chưa có blueprint. Nhấn 'Tạo blueprint từ ma trận'.")
        return

    st.markdown("### Sinh câu hỏi")
    max_tasks = 60
    if len(st.session_state.blueprint) > max_tasks:
        st.warning(f"Blueprint hiện có {len(st.session_state.blueprint)} task. V1.1 khuyến nghị test <= {max_tasks} để tránh quá tải.")

    colA, colB, colC = st.columns([1, 1, 2])
    only_first_n = colA.number_input("Sinh tối đa", min_value=1, max_value=max(len(st.session_state.blueprint), 1), value=min(len(st.session_state.blueprint), 30))
    do_overwrite = colB.checkbox("Ghi đè câu cũ", value=False)

    if colC.button("✨ TẠO ĐỀ", disabled=st.session_state.matrix_locked is False):
        st.warning("Bạn nên CHỐT ma trận trước khi tạo đề (để giữ form).")

    if st.button("✨ TẠO ĐỀ (theo blueprint)"):
        if do_overwrite:
            st.session_state.questions = []
        prog = st.progress(0)
        questions = list(st.session_state.questions)
        blueprint = st.session_state.blueprint[: int(only_first_n)]
        for i, tdict in enumerate(blueprint, start=1):
            from modules.matrix_engine import Task

            task = Task(**tdict)
            q = generate_question(
                task,
                api_key=cfg["api_key"],
                model=cfg["model"],
                api_base=cfg["api_base"],
                temperature=cfg["temperature"],
                max_output_tokens=cfg["max_output_tokens"],
            )
            questions.append(q)
            prog.progress(i / len(blueprint))
        st.session_state.questions = questions
        st.success(f"Đã sinh {len(blueprint)} câu (tổng hiện có: {len(questions)})")

    if st.button("🔁 TẠO LẠI ĐỀ (giữ form)"):
        prog = st.progress(0)
        new_questions = []
        blueprint = st.session_state.blueprint
        for i, tdict in enumerate(blueprint, start=1):
            from modules.matrix_engine import Task

            task = Task(**tdict)
            q = generate_question(
                task,
                api_key=cfg["api_key"],
                model=cfg["model"],
                api_base=cfg["api_base"],
                temperature=cfg["temperature"],
                max_output_tokens=cfg["max_output_tokens"],
            )
            new_questions.append(q)
            prog.progress(i / len(blueprint))
        st.session_state.questions = new_questions
        st.success(f"Đã tạo lại {len(new_questions)} câu")

    st.divider()

    st.markdown("### Danh sách câu hỏi")

    questions = st.session_state.questions
    if not questions:
        st.info("Chưa có câu hỏi. Nhấn 'TẠO ĐỀ'.")
        return

    # Validate total points
    _e, _w = validate_exam_points(questions)
    for w in _w:
        st.warning(w)

    # Render questions
    for idx, q in enumerate(questions, start=1):
        errs, warns = validate_question(q)
        title = f"Câu {idx} – {q.get('qtype','')} ({q.get('level','')}, {q.get('points','')}đ)"
        with st.expander(title, expanded=False):
            if errs:
                st.error("; ".join(errs))
            if warns:
                st.warning("; ".join(warns))

            # simple editor per type
            qt = q.get("qtype")
            content = q.get("content") or {}

            if qt == "Trắc nghiệm nhiều lựa chọn":
                content["stem"] = st.text_area("Nội dung", value=content.get("stem", ""), key=f"stem_{idx}")
                opts = content.get("options") or {"A": "", "B": "", "C": "", "D": ""}
                for k in ["A", "B", "C", "D"]:
                    opts[k] = st.text_input(f"{k}", value=opts.get(k, ""), key=f"opt_{idx}_{k}")
                content["options"] = opts
                content["answer"] = st.selectbox("Đáp án", ["A", "B", "C", "D"], index=["A","B","C","D"].index(str(content.get("answer","A")).upper()), key=f"ans_{idx}")

            elif qt == "Đúng/Sai":
                content["stem"] = st.text_input("Hướng dẫn", value=content.get("stem", ""), key=f"tf_stem_{idx}")
                statements = content.get("statements") or []
                # fixed 4 for now
                while len(statements) < 4:
                    statements.append({"text": "", "answer": True})
                new_st = []
                for j in range(4):
                    c1, c2 = st.columns([4, 1])
                    txt = c1.text_input(f"Mệnh đề {j+1}", value=statements[j].get("text", ""), key=f"tf_{idx}_{j}")
                    ans = c2.selectbox("Đ/S", ["Đ", "S"], index=0 if statements[j].get("answer") in [True, "Đ", "Đúng"] else 1, key=f"tf_ans_{idx}_{j}")
                    new_st.append({"text": txt, "answer": True if ans == "Đ" else False})
                content["statements"] = new_st

            elif qt == "Nối cột":
                content["stem"] = st.text_input("Hướng dẫn", value=content.get("stem", ""), key=f"match_stem_{idx}")
                left = content.get("left") or ["", "", "", ""]
                right = content.get("right") or ["", "", "", ""]
                n = st.number_input("Số cặp", min_value=2, max_value=8, value=max(2, min(4, len(left))), key=f"match_n_{idx}")
                left = (left + [""] * 8)[:n]
                right = (right + [""] * 8)[:n]
                st.write("Cột A")
                for j in range(n):
                    left[j] = st.text_input(f"A{j+1}", value=left[j], key=f"match_a_{idx}_{j}")
                st.write("Cột B")
                for j in range(n):
                    right[j] = st.text_input(f"B{j+1}", value=right[j], key=f"match_b_{idx}_{j}")
                content["left"] = left
                content["right"] = right
                # mapping editor as JSON
                mapping_text = st.text_area("Mapping đáp án (JSON)", value=json.dumps(content.get("mapping", {}), ensure_ascii=False, indent=2), key=f"match_map_{idx}")
                try:
                    content["mapping"] = json.loads(mapping_text)
                except Exception:
                    st.warning("Mapping JSON không hợp lệ. Giữ mapping cũ.")

            elif qt == "Điền khuyết":
                content["stem"] = st.text_input("Hướng dẫn", value=content.get("stem", ""), key=f"blank_stem_{idx}")
                content["text"] = st.text_area("Văn bản", value=content.get("text", ""), key=f"blank_text_{idx}")
                content["answer"] = st.text_input("Đáp án", value=str(content.get("answer", "")), key=f"blank_ans_{idx}")

            elif qt == "Tự luận":
                content["prompt"] = st.text_area("Đề bài", value=content.get("prompt", ""), key=f"essay_p_{idx}")
                rubric = content.get("rubric") or []
                rubric_text = "\n".join(str(x) for x in rubric)
                rubric_text = st.text_area("Rubric (mỗi dòng 1 ý)", value=rubric_text, key=f"essay_r_{idx}")
                content["rubric"] = [line.strip() for line in rubric_text.splitlines() if line.strip()]

            q["content"] = content

            # controls
            b1, b2, b3 = st.columns([1, 1, 3])
            if b1.button("🔁 Tạo lại câu này", key=f"regen_{idx}"):
                from modules.matrix_engine import Task

                # find task by task_id
                tid = q.get("task_id")
                tdict = next((t for t in st.session_state.blueprint if t.get("task_id") == tid), None)
                if tdict:
                    task = Task(**tdict)
                    new_q = generate_question(
                        task,
                        api_key=cfg["api_key"],
                        model=cfg["model"],
                        api_base=cfg["api_base"],
                        temperature=cfg["temperature"],
                        max_output_tokens=cfg["max_output_tokens"],
                    )
                    questions[idx - 1] = new_q
                    st.session_state.questions = questions
                    st.success("Đã tạo lại câu")
                else:
                    st.error("Không tìm thấy task tương ứng")

            if b2.button("🗑️ Xoá", key=f"del_{idx}"):
                questions.pop(idx - 1)
                st.session_state.questions = questions
                st.experimental_rerun()

    st.session_state.questions = questions


def _tab_export():
    st.header("Tab 3 – Tải xuống")

    matrix = normalize_matrix(st.session_state.matrix)
    questions = st.session_state.questions
    meta = st.session_state.meta

    if matrix.empty:
        st.info("Chưa có ma trận")
        return

    st.markdown("### Export")
    col1, col2, col3 = st.columns([1, 1, 1])

    # Matrix Excel
    if col1.button("📥 Tải ma trận Excel"):
        bio = export_matrix_to_excel(matrix.fillna("").to_dict(orient="records"), template_path=PATHS.repo_root / "data" / "matrices" / "templates" / "matrix_template.xlsx")
        st.download_button("Download ma trận.xlsx", data=bio, file_name="ma_tran.xlsx")

    # Blueprint Word
    if col2.button("📥 Tải Bảng đặc tả Word"):
        bio = build_blueprint_docx(matrix.fillna("").to_dict(orient="records"), meta=meta, template_bytes=st.session_state.tpl_blueprint)
        st.download_button("Download bang_dac_ta.docx", data=bio, file_name="bang_dac_ta.docx")

    # Exam Word
    if col3.button("📥 Tải Đề Word"):
        if not questions:
            st.warning("Chưa có câu hỏi để xuất đề")
        else:
            # subject auto
            if not meta.get("subject"):
                subjects = sorted(set(matrix["Môn"].dropna().tolist()))
                if len(subjects) == 1:
                    meta["subject"] = subjects[0]
            bio = build_exam_docx(questions, meta=meta, template_bytes=st.session_state.tpl_exam)
            st.download_button("Download de.docx", data=bio, file_name="de.docx")

    st.divider()

    st.markdown("### Export session JSON")
    sess_json = export_session_json(
        matrix_df=matrix,
        blueprint=list(st.session_state.blueprint),
        questions=list(st.session_state.questions),
        meta=meta,
    )
    st.download_button("Download session.json", data=sess_json.encode("utf-8"), file_name="session.json")


def main():
    st.set_page_config(page_title="de-maker-grade5 V1.1", layout="wide")
    _init_state()
    _load_default_data()

    cfg = _sidebar()

    st.title("de-maker-grade5 – V1.1")
    st.caption("Test pipeline: Ma trận → Sinh câu hỏi → Xuất Word")

    tab1, tab2, tab3 = st.tabs(["1) Ma trận", "2) Tạo câu hỏi", "3) Tải xuống"])

    with tab1:
        _tab_matrix()
    with tab2:
        _tab_questions(cfg)
    with tab3:
        _tab_export()


if __name__ == "__main__":
    main()
