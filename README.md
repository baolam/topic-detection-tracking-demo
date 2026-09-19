# Dynamic Topic Detection and Tracking System (Demo)

Hệ thống Phát hiện và Theo dõi Chủ đề Động trên Luồng Dữ liệu Mạng Xã hội áp dụng thuật toán **DenStream** tích hợp **Cơ chế Lãng quên (Exponential Time-Decay)** và **Real-Time Text Embeddings**.

---

## 📂 Cấu trúc Dự án

```
topic-detection-tracking/
├── .venv/                      # Môi trường ảo Python riêng biệt
├── config/                     # Cấu hình tham số (Lambda, Epsilon, Mu)
│   └── config.yaml             
├── src/                        # Mã nguồn chính
│   ├── data/                   # Data Schema & Real-time Stream Simulator
│   ├── embeddings/             # Real-time Dense Vector Embedder (MiniLM / Hash)
│   ├── engine/                 # Động cơ DenStream + Time-Decay + Eviction Policy
│   ├── tracking/               # Macro-Clustering (DBSCAN) & Topic Labeler
│   └── utils/                  # Logger utilities
├── tests/                      # Unit tests cho Time-decay toán học
│   └── test_denstream.py       
├── main.py                     # CLI Runner chạy demo trực tiếp trên Terminal
├── app.py                      # Interactive Streamlit Dashboard
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
.\.venv\Scripts\python.exe tests/test_denstream.py
```

### 4. Chạy Demo trên Terminal (CLI Runner)
```powershell
.\.venv\Scripts\python.exe main.py
```

### 5. Khởi Chạy Giao Diện Web Dashboard (Streamlit UI)
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

---

## 🔬 Góc Nhìn Dữ Liệu (Data Perspective)
1. **Input Schema**: `SocialPost` chứa `post_id`, `timestamp`, `text`, `author_id`, `hashtags`.
2. **Exponential Decay Math**: Trọng số $w(t) = 2^{-\lambda (t - t_0)}$. Sau khoảng $T_{half-life} = \frac{1}{\lambda}$, bài đăng giảm 50% độ nóng.
3. **Density-based Outlier Promotion**: Outlier Micro-Cluster ($o$-MC) khi có đợt bài đăng bùng nổ sẽ tích lũy trọng số vượt ngưỡng $\beta \cdot \mu$ và biến thành cụm tiềm năng ($p$-MC) đại diện cho **Breaking Event**.
