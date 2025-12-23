# de-maker-grade5 (V1.1)

Tool Streamlit hỗ trợ tạo **ma trận → sinh câu hỏi → ghép đề → xuất Word** cho **lớp 5**.

V1.1 tập trung để bạn **test dữ liệu lớp 5** (kho YCCĐ + PPCT/số tiết + ma trận) và kiểm tra quy trình:
- Chọn dữ liệu theo dòng ma trận
- Nút **Tạo đề** / **Tạo lại đề** (giữ nguyên form: dạng × mức × điểm)
- Preview, sửa, xoá, tạo lại từng câu
- Xuất **Đề (docx)** + **Bảng đặc tả (docx)** + **Ma trận Excel**

## Chạy thử

```bash
# 1) Tạo môi trường
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Cài
pip install -r requirements.txt

# 3) Chạy
streamlit run app.py
```

## Dùng Gemini API (AI Studio key)

App gọi **Gemini Developer API** qua REST `generateContent` (header `x-goog-api-key`).

### Cách 1: nhập key trên sidebar
- Mở app → sidebar → dán `Gemini API key`.

### Cách 2: dùng Streamlit Secrets / biến môi trường
- Local: tạo file `.streamlit/secrets.toml` dựa trên `.streamlit/secrets.toml.example`.
- Streamlit Cloud: Settings → Secrets → thêm:

```toml
GEMINI_API_KEY = "..."
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
```

> Không có key: app vẫn chạy được để test pipeline (sinh câu hỏi theo template offline).

## Deploy lên GitHub + Streamlit Cloud

1) Đẩy repo lên GitHub (public/private đều được)
2) Vào Streamlit Community Cloud → New app
3) Chọn repo + branch + file chạy `app.py`
4) Thêm Secrets (như phần trên) → Deploy

## Dữ liệu mẫu kèm repo
- `data/yccd/khoi5_normalized.xlsx` và `.csv`
- `data/matrices/templates/matrix_template.xlsx` (mẫu ma trận)
- `data/matrices/templates/example_*.xlsx` (2 mẫu công thức tỉ lệ/điểm)
- `data/ppct/ppct_grade5_master.csv` (hiện kèm Toán HK1 từ dữ liệu bạn đã gửi; các môn khác có thể bổ sung)

## Gợi ý bổ sung PPCT
Bạn có thể:
- Upload thêm PPCT/Excel ngay trong app, hoặc
- Chạy script `scripts/build_ppct_from_k5_pdf.py` để trích số tiết sơ bộ từ `data/ppct/sources/K5.pdf` (cần kiểm tra lại vì PDF có thể tách dòng).

## Template Word
- V1.1 có thể xuất Word theo layout mặc định (chuẩn NĐ30 cơ bản).
- Nếu bạn muốn khớp đúng mẫu trường, upload mẫu `.docx` trong sidebar để app thử điền dữ liệu vào bảng.

> Lưu ý: python-docx không đọc `.doc`. Hãy dùng `.docx`.

