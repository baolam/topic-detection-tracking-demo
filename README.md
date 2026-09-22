# Dynamic Topic Detection and Tracking System (SADStream Demo)

Hệ thống Phát hiện và Theo dõi Chủ đề Động trên Luồng Dữ liệu Mạng Xã hội áp dụng thuật toán đề xuất **SADStream (Adaptive Semantic Stream Clustering)** tích hợp **Tốc độ Suy giảm Thích ứng ($\lambda_k(t)$)**, **Bộ nhớ Cụm Nén Dài hạn (Compressed Prototype Memory)** và **Độ tương đồng Hỗn hợp (Hybrid Similarity)**.

---

## 📂 Cấu trúc Dự án

```
topic-detection-tracking/
├── .venv/                      # Môi trường ảo Python riêng biệt
├── config/                     # Cấu hình tham số (Lambda, Epsilon, Mu, Hybrid Weights)
│   └── config.yaml             
├── src/                        # Mã nguồn chính
│   ├── data/                   # Data Schema & Real-time Stream Simulator (CrisisBench / Synthetic)
│   ├── embeddings/             # Real-time Dense Vector Embedder (all-MiniLM-L6-v2)
│   ├── engine/                 # SADStream Engine (Adaptive Decay + Memory Archive) & DenStream Baseline
│   ├── tracking/               # Macro-Clustering (DBSCAN) & Topic Labeler (c-TF-IDF)
│   ├── evaluation/             # Evaluator & Purity Metrics (NMI, ARI, Purity)
│   └── ui/                     # Streamlit Dashboard UI (Components, Styles)
├── tests/                      # Unit tests kiểm thử thuật toán
│   ├── test_denstream.py       
│   └── test_sadstream.py       # Kiểm thử SADStream Adaptive Decay & Memory Archive
├── main.py                     # CLI Runner chạy demo trực tiếp trên Terminal
├── eval.py                     # Script Đánh giá So sánh Benchmark (SADStream vs DenStream)
├── app.py                      # Interactive Streamlit Web Dashboard
└── requirements.txt            # Danh sách gói thư viện
```

---

## 🚀 Hướng Dẫn Cài Đặt & Vận Hành

### 1. Kích hoạt Môi trường Ảo (Virtual Environment)
Môi trường ảo đã được tạo sẵn tại thư mục `.venv`:
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 2. Cài Đặt Dependencies (Nếu cần)
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 3. Chạy Kiểm Thử Đơn Vị (Unit Tests)
```powershell
.\.venv\Scripts\python.exe -m pytest
```

### 4. Chạy Benchmark So Sánh (Evaluator)
```powershell
.\.venv\Scripts\python.exe eval.py
```

### 5. Khởi Chạy Giao Diện Web Dashboard (Streamlit UI)
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

---

## 🔬 Thuật Toán SADStream (SADStream Algorithm Features)
1. **Adaptive Decay Rate ($\lambda_k(t)$)**: Tự động điều chỉnh tốc độ suy giảm thời gian dựa trên độ bùng nổ ($B_k$) và mức độ hoạt động ($A_k$) của từng cụm topic. Các chủ đề đang bùng nổ sẽ được duy trì độ nóng lâu hơn.
2. **Compressed Prototype Memory ($L_k$)**: Tự động nén và lưu trữ các cụm bị loại bỏ vào kho lưu trữ bộ nhớ dài hạn, cho phép phục hồi chủ đề lặp lại (Recurring Topics) khi có bài đăng mới tương đồng.
3. **Hybrid Text-Feature Similarity**: Kết hợp độ tương đồng Semantic Cosine ($S_{sem}$), Lexical Jaccard ($S_{lex}$), Entity Jaccard ($S_{ent}$) và Hashtag Jaccard ($S_{hash}$).
4. **Interactive Web Demo Control**: Giao diện Dashboard Web (`app.py`) cho phép tùy chỉnh trực tiếp bộ tham số SADStream, xem trực quan hóa 2D/3D PCA vector space kèm các cụm bộ nhớ nén (Compressed Memory Prototypes).

