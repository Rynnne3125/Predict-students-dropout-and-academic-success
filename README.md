# ML-EduPredict — Dự đoán Bỏ học & Kết quả Học tập Sinh viên

Dự án **Học máy (Machine Learning)** phân loại đa lớp nhằm dự đoán kết quả học tập của sinh viên đại học: **Bỏ học (Dropout)**, **Đang học (Enrolled)** hoặc **Tốt nghiệp (Graduate)**. Hệ thống gồm quy trình phân tích dữ liệu (Jupyter Notebook), script huấn luyện mô hình production và **ứng dụng web Streamlit** đầy đủ tính năng demo / triển khai.


---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Bài toán & mục tiêu](#bài-toán--mục-tiêu)
3. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
4. [Bộ dữ liệu](#bộ-dữ-liệu)
5. [Quy trình xử lý & mô hình](#quy-trình-xử-lý--mô-hình)
6. [Cài đặt môi trường](#cài-đặt-môi-trường)
7. [Hướng dẫn sử dụng nhanh](#hướng-dẫn-sử-dụng-nhanh)
8. [Ứng dụng web Streamlit (`app.py`)](#ứng-dụng-web-streamlit-apppy)
9. [Kịch bản demo (`Demo Scenario.text`)](#kịch-bản-demo-demo-scenariotext)
10. [Jupyter Notebook](#jupyter-notebook-phân-tích--thử-nghiệm)
11. [Mô hình đã lưu (`model.pkl`)](#mô-hình-đã-lưu-modelpkl)
12. [Lưu ý kỹ thuật & xử lý sự cố](#lưu-ý-kỹ-thuật--xử-lý-sự-cố)
13. [Tác giả](#tác-giả)

---

## Tổng quan

| Thành phần | Mô tả |
|------------|--------|
| **Dữ liệu** | `dataset.csv` — 4.424 sinh viên, 36 đặc trưng gốc + nhãn `Target` |
| **Notebook** | EDA, feature engineering, PCA, K-Means, SMOTE, so sánh 7 thuật toán |
| **Huấn luyện** | `train_and_save.py` — Scale → SMOTE → **Voting Ensemble** (XGB + LGBM + RF) |
| **Triển khai** | `app.py` — Streamlit: dự đoán, dashboard, lịch sử, báo cáo HTML |
| **Đầu ra** | `model.pkl` (~46 MB) — ensemble + scaler + label encoder + `feature_cols` |
| **Demo** | `Demo Scenario.text` — kịch bản trình bày 5 ca + timeline 12/25 phút |

---

## Bài toán & mục tiêu

- **Loại bài toán:** Phân loại đa lớp (3 nhãn).
- **Mục tiêu nghiệp vụ:** Nhận diện sớm sinh viên có nguy cơ bỏ học để can thiệp (học phí, cố vấn, tutoring).

| Nhãn | Ý nghĩa | Tỷ lệ (dataset) |
|------|---------|-----------------|
| `Graduate` | Tốt nghiệp | ~49,9% |
| `Dropout` | Bỏ học | ~32,1% |
| `Enrolled` | Đang theo học | ~17,1% |

Nguồn dữ liệu: bộ công khai *Predict students' dropout and academic success* (bối cảnh giáo dục đại học Bồ Đào Nha).

---

## Cấu trúc thư mục

```
ML2026/
├── dataset.csv                    # Dữ liệu gốc (sep ;)
├── train_and_save.py              # Huấn luyện → model.pkl
├── app.py                         # Ứng dụng Streamlit (UI + predict)
├── model.pkl                      # Pipeline đã train (không commit Git)
├── prediction_history.json        # Lịch sử dự đoán local (tự tạo, không commit)
├── Demo Scenario.text             # Kịch bản demo chi tiết cho hội đồng
├── requirements.txt               # Dependencies Python
├── .gitignore                     # venv/, model.pkl, prediction_history.json, ...
├── README.md                      # Tài liệu này
├── Predict students' dropout and academic success.ipynb
├── Predict students' dropout and academic success.html
└── venv/                          # Virtual environment (không commit)
```

---

## Bộ dữ liệu

| Thuộc tính | Giá trị |
|------------|---------|
| File | `dataset.csv` |
| Phân cách | `;` |
| Kích thước | 4.424 × 37 (36 feature + `Target`) |
| Missing / duplicate | Không |

Sau khi đọc, script chuẩn hóa tên cột: khoảng trắng → `_` (ví dụ `Marital status` → `Marital_status`).

### Nhóm đặc trư (36 cột gốc)

| Nhóm | Ví dụ cột |
|------|-----------|
| Hành chính / tuyển sinh | `Marital_status`, `Application_mode`, `Course`, `Admission_grade` |
| Gia đình / xã hội | `Mother's_qualification`, `Father's_occupation`, … |
| Sinh viên | `Debtor`, `Tuition_fees_up_to_date`, `Gender`, `Age_at_enrollment`, … |
| Học kỳ 1 & 2 | `Curricular_units_1st_sem_(enrolled/approved/grade/…)` |
| Kinh tế vĩ mô | `Unemployment_rate`, `Inflation_rate`, `GDP` |

### Feature engineering (bắt buộc khi predict)

| Cột mới | Công thức |
|---------|-----------|
| `pass_rate_1` | `approved_1 / (enrolled_1 + 1)` |
| `pass_rate_2` | `approved_2 / (enrolled_2 + 1)` |
| `grade_progress` | `grade_2 − grade_1` |
| `total_approved` | `approved_1 + approved_2` |
| `avg_grade` | `(grade_1 + grade_2) / 2` |

**Tổng cột đầu vào mô hình:** 41 (36 + 5).  
`app.py` **phải** tính 5 cột này trước `scaler.transform` — nếu thiếu, mọi ca có thể bị dự đoán sai (ví dụ luôn Dropout).

---

## Quy trình xử lý & mô hình

### Pipeline production (`train_and_save.py`)

```
dataset.csv (;)
  → Chuẩn hóa tên cột
  → Feature Engineering (5 cột)
  → LabelEncoder(Target)
  → Train/Test 80/20 (random_state=42)
  → StandardScaler (fit train)
  → SMOTE (chỉ train)
  → VotingClassifier (soft, weights [3,2,1])
        ├── XGBoost
        ├── LightGBM
        └── RandomForest
  → Đánh giá test + lưu model.pkl
```

**Soft voting:** trung bình có trọng số xác suất 3 mô hình con.

**Thời gian train:** khoảng 3–5 phút (CPU).

### Pipeline nghiên cứu (Notebook)

EDA → IQR outlier → Chi-square → **PCA (10)** + **K-Means (3)** → SMOTE → so sánh 7 classifier.  
Kết luận notebook: XGBoost mạnh trên pipeline PCA; **production** dùng full 41 feature + ensemble để deploy ổn định hơn.

---

## Cài đặt môi trường

**Yêu cầu:** Python 3.9+, RAM ≥ 8 GB khi train.

```bash
cd /đường/dẫn/ML2026
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Hoặc cài tay:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm streamlit altair
```

*(Notebook thêm: `matplotlib`, `seaborn`, `jupyter`, `catboost` nếu cần.)*

---

## Hướng dẫn sử dụng nhanh
# Chạy ứng dụng
streamlit run app.py
# → http://localhost:8501
```

## Ứng dụng web Streamlit (`app.py`)

