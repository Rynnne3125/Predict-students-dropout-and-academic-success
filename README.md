Predict-students-dropout-and-academic-success

Dự án **Machine Learning** nhằm dự đoán kết quả học tập của sinh viên với 3 nhãn chính: **Dropout**, **Enrolled** và **Graduate**. Phiên bản hiện tại tập trung vào **demo và triển khai giao diện Streamlit**, sử dụng các mô hình đã huấn luyện sẵn và dữ liệu minh họa để trình bày kết quả một cách trực quan.

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Cấu trúc dự án hiện tại](#cấu-trúc-dự-án-hiện-tại)
3. [Bộ dữ liệu](#bộ-dữ-liệu)
4. [Ứng dụng Streamlit](#ứng-dụng-streamlit)
5. [Cài đặt môi trường](#cài-đặt-môi-trường)
6. [Hướng dẫn chạy dự án](#hướng-dẫn-chạy-dự-án)
7. [Notebook phân tích](#notebook-phân-tích)
8. [Mô hình và artifact hiện tại](#mô-hình-và-artifact-hiện-tại)
9. [Ghi chú triển khai](#ghi-chú-triển-khai)

---

## Tổng quan

- **Bài toán:** phân loại đa lớp.
- **Mục tiêu:** hỗ trợ nhận diện sinh viên có nguy cơ bỏ học, đồng thời cung cấp trực quan hóa kết quả và khả năng đối chiếu với hồ sơ mẫu.
- **Công nghệ chính:** `Python`, `pandas`, `scikit-learn`, `xgboost`, `streamlit`, `matplotlib`, `seaborn`.

---

## Cấu trúc dự án hiện tại

```text
ML-EduPredict/
├── app.py                    # Ứng dụng Streamlit chính
├── dataset.csv               # Dữ liệu gốc của bài toán
├── demo_profiles.csv         # Hồ sơ mẫu dùng cho demo
├── requirements.txt          # Danh sách thư viện cần cài
├── scaler.pkl                # Artifacts đã huấn luyện
├── pca.pkl                   # Artifacts đã huấn luyện
├── kmeans.pkl                # Artifacts đã huấn luyện
├── best_student_dropout_xgb.pkl  # Mô hình XGBoost được tải trong app
├── Predict students' dropout and academic success.ipynb
└── README.md                 # Tài liệu dự án
```

### File quan trọng

- `app.py`: chạy dashboard Streamlit với 3 tab:
  - dự đoán và đối chiếu với hồ sơ mẫu
  - bảng so sánh hiệu năng đa mô hình
  - tổng quan dữ liệu và nhật ký cải tiến pipeline
- `dataset.csv`: dữ liệu chính, phân cách bằng `;`
- `demo_profiles.csv`: dữ liệu minh họa để thử nghiệm giao diện
- `requirements.txt`: danh sách dependency sử dụng trong dự án
- `best_student_dropout_xgb.pkl`, `scaler.pkl`, `pca.pkl`, `kmeans.pkl`: artifact mô hình dùng trực tiếp trong `app.py`

---

## Bộ dữ liệu

### `dataset.csv`

- **Nguồn dữ liệu:** bộ dữ liệu công khai về sinh viên đại học.
- **Phân cách:** dấu `;`
- **Kích thước:** 4.424 dòng và 37 cột (bao gồm nhãn `Target`).
- **Cột chính:** thông tin hành chính, học tập, tài chính, gia đình và các chỉ số kinh tế vĩ mô.

### `demo_profiles.csv`

- Chứa **3 hồ sơ minh họa** được dùng để kiểm thử giao diện.
- Mỗi hồ sơ tương ứng với một nhãn thực tế:
  - `Dropout`
  - `Enrolled`
  - `Graduate`

---

## Ứng dụng Streamlit

`app.py` hiện tại là phiên bản **dashboard chuyên nghiệp** với các chức năng sau:

1. **Dự đoán và đối chiếu thực tế**
   - người dùng chọn một hồ sơ demo
   - chỉnh sửa các biến cốt lõi trên giao diện
   - xem dự đoán của mô hình và so sánh với nhãn thực tế

2. **Bảng so sánh hiệu năng đa mô hình**
   - hiển thị độ chính xác của các thuật toán đã thử nghiệm trong notebook

3. **Tổng quan dữ liệu và nhật ký cải tiến pipeline**
   - mô tả dữ liệu, các bước tiền xử lý và lý do chọn mô hình hiện tại

### Luồng xử lý trong app

- Tải các artifact `.pkl`
- Đọc `demo_profiles.csv`
- Ghi đè các giá trị từ giao diện vào hồ sơ mẫu
- Chuẩn hóa dữ liệu
- Áp dụng `PCA`
- Áp dụng `KMeans`
- Dự đoán bằng `XGBoost`

---

## Cài đặt môi trường

### Yêu cầu

- Python 3.9+
- Có thể dùng `venv` hoặc `conda`

### Cài đặt

```bash
cd D:\Documents\ML\ML-EduPredict
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Nếu muốn cài thủ công

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm streamlit altair matplotlib seaborn
```

---

## Hướng dẫn chạy dự án

### Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Sau đó mở trình duyệt tại địa chỉ:

```text
http://localhost:8501
```

### Kiểm tra nhanh các file mô hình

Trước khi chạy `app.py`, hãy đảm bảo các file sau tồn tại trong thư mục dự án:

- `scaler.pkl`
- `pca.pkl`
- `kmeans.pkl`
- `best_student_dropout_xgb.pkl`

Nếu thiếu bất kỳ file nào, hãy mở notebook để tạo lại các artifact hoặc bổ sung thủ công.

---

## Notebook phân tích

File:

- `Predict students' dropout and academic success.ipynb`

Notebook này là nơi thực hiện:

- phân tích exploratory data analysis
- so sánh nhiều thuật toán
- kiểm thử pipeline tiền xử lý
- đánh giá hiệu năng mô hình
- xuất các file `.pkl` dùng cho ứng dụng

---

## Mô hình và artifact hiện tại

Phiên bản hiện tại của dự án đang sử dụng các artifact sau để phục vụ giao diện:

| File | Vai trò |
| --- | --- |
| `best_student_dropout_xgb.pkl` | Mô hình XGBoost dùng cho dự đoán |
| `scaler.pkl` | Bộ chuẩn hóa dữ liệu đầu vào |
| `pca.pkl` | Bộ giảm chiều PCA |
| `kmeans.pkl` | Mô hình phân cụm |

Ứng dụng tải toàn bộ các file này trong `app.py` để tạo một pipeline inference hoàn chỉnh.

---

## Ghi chú triển khai

- README này được viết để **phù hợp với cấu trúc hiện tại của dự án**.
- `app.py` đang chạy theo thiết kế **demo + inference**, không phải theo cấu trúc `train_and_save.py` cũ.
- Nếu bạn muốn, tôi có thể tiếp tục:
  1. bổ sung phần **cách tái huấn luyện mô hình** vào README
  2. chuẩn hóa lại cấu trúc thư mục theo chuẩn `src/`
  3. thêm **hướng dẫn Docker / deploy**
