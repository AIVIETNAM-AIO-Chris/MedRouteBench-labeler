# MedRouteBench: CheXpert Labeler Pipeline for MIMIC-CXR

Hệ thống xử lý và gán nhãn tự động **14 bệnh lý lồng ngực CheXpert** cho tập dữ liệu đa nghiên cứu **MIMIC-CXR VQA (`mimic_cxr_aug_train.csv`)** dựa trên công cụ chuẩn của Đại học Stanford (*Irvin et al., 2019*).

---

## 📌 Tính năng nổi bật

- **Xử lý cấu trúc đa report ($N$ reports/sample):** Bóc tách các chuỗi danh sách Python literal chứa nhiều báo cáo bệnh án trên cùng một bệnh nhân/lần khám.
- **Khử trùng lặp (Deduplication):** Tự động lọc các report trống/tầm thường (`Findings: Impression:`) và gộp các report giống nhau, giúp tiết kiệm đáng kể thời gian chạy parser ngữ pháp.
- **Ánh xạ nhãn 2 cấp độ:**
  1. **Cấp độ Sample (14 cột bệnh lý):** Tổng hợp bằng thuật toán *Hierarchical Max-Pooling* y khoa ($1.0 > -1.0 > 0.0 > \text{NaN}$).
  2. **Cấp độ Report (cột `report_labels`):** Lưu dạng chuỗi JSON danh sách nhãn riêng của từng report, phục vụ chính xác cho bài toán **VQA (Visual Question Answering)** khi cần gióng nhãn từng ảnh với từng report.
- **Hỗ trợ 2 chế độ linh hoạt:**
  - 🐳 **Docker (Khuyên dùng):** Đóng gói toàn bộ môi trường C++ / Java / Python 3.7 chuẩn xác, không sợ lỗi phụ thuộc.
  - 🐍 **Native / Conda (`--no_docker`):** Chạy trực tiếp trên máy chủ/server tính toán không hỗ trợ hoặc không có quyền cài Docker.

---

## 📂 Cấu trúc thư mục

```text
MedRouteBench/
├── chexpert-labeler/              # Mã nguồn công cụ Stanford CheXpert Labeler
│   ├── Dockerfile                 # Cấu hình Docker tối ưu (Python 3.7 + Java + pinned libs)
│   ├── label.py                   # Script gắn nhãn gốc của Stanford
│   ├── stages/                    # Các giai đoạn: extract, classify, aggregate
│   └── phrases/ & patterns/       # Từ điển bệnh lý & luật phủ định (negation)
├── run_pipeline.py                # Script điều phối toàn bộ luồng (End-to-End Orchestrator)
├── prepare_chexpert_input.py      # Bóc tách, deduplicate và tạo file input 1 cột cho CheXpert
├── aggregate_chexpert_labels.py   # Ghép nhãn trở lại sample và tổng hợp hierarchical pooling
├── read_csv_data.py               # Module đọc CSV & giải mã danh sách an toàn
├── requirements.txt               # Danh sách các gói Python và phiên bản tương thích
├── mimic_cxr_aug_train.csv        # File dữ liệu đầu vào (đặt tại đây)
└── data_work/                     # Thư mục lưu dữ liệu trung gian và kết quả đầu ra (tự tạo)
    ├── chexpert_input.csv         # File danh sách report duy nhất nạp cho tool
    ├── mapping_metadata.json      # Bản đồ ánh xạ giữa sample và report
    ├── chexpert_output.csv        # Kết quả nhãn thô từ CheXpert
    └── mimic_cxr_aug_train_labeled.csv  # KẾT QUẢ CUỐI CÙNG HOÀN CHỈNH
```

---

## 🛠️ Hướng dẫn cài đặt

### Yêu cầu tiên quyết:
1. Đặt file dữ liệu **`mimic_cxr_aug_train.csv`** vào thư mục gốc của dự án.
2. Chọn một trong hai cách cài đặt bên dưới:

---

### Cách 1: Sử dụng Docker (Khuyên dùng - Nhanh & Ổn định nhất)

Nếu máy bạn hoặc server đã cài **Docker / Docker Desktop**:

1. **Cài đặt thư viện điều phối ở máy chủ (Host):**
   ```bash
   pip install pandas tqdm
   ```

2. **Build Docker Image:**
   ```bash
   docker build -t chexpert-labeler:latest ./chexpert-labeler
   ```
   *(Kiểm tra lệnh `docker images` thấy có `chexpert-labeler:latest` là hoàn tất).*

---

### Cách 2: Sử dụng Conda (Dành cho máy/server KHÔNG CÓ DOCKER)

Nếu server không có Docker, hãy thiết lập môi trường Conda với Python 3.7 và Java Runtime:

1. **Tạo và kích hoạt môi trường Conda:**
   ```bash
   conda create -n chexpert python=3.7 openjdk git pip -y
   conda activate chexpert
   ```

2. **Tải mã nguồn NegBio vào thư mục tool:**
   ```bash
   cd chexpert-labeler
   git clone https://github.com/ncbi-nlp/NegBio.git
   cd ..
   ```

3. **Cài đặt các gói thư viện từ `requirements.txt`:**
   ```bash
   python -m pip install -r requirements.txt
   ```
   *(Đặc biệt lưu ý: `networkx==2.3` và `bioc==1.0` phải đúng phiên bản ghim sẵn).*

4. **Tải các tập dữ liệu NLTK và model phân tích GENIA:**
   ```bash
   python -m nltk.downloader universal_tagset punkt wordnet omw-1.4
   python -c "from bllipparser import RerankingParser; RerankingParser.fetch_and_load('GENIA+PubMed')"
   ```

---

## 🚀 Hướng dẫn chạy Pipeline

Script `run_pipeline.py` điều phối toàn bộ quá trình tự động từ chuẩn bị dữ liệu, gọi tool gán nhãn, đến tổng hợp nhãn cuối cùng.

### 1. Chạy thử nghiệm nhanh (Smoke Test trên 20 mẫu)
Nên chạy thử trên 20 mẫu để kiểm tra kết quả trước khi chạy toàn bộ:

- **Nếu dùng Docker:**
  ```bash
  python run_pipeline.py --step all --nrows 20
  ```

- **Nếu không dùng Docker (Local Conda):**
  ```bash
  python run_pipeline.py --step all --no_docker --nrows 20
  ```

### 2. Chạy chính thức toàn bộ tập dữ liệu (Full Dataset - 64,586 mẫu)

#### ⚡ Tăng tốc đa tiến trình (Khuyên dùng trên Server đa nhân):
Sử dụng cờ `--workers <N>` (ví dụ `--workers 16` hoặc `--workers 0` để tự động tận dụng toàn bộ số nhân CPU):

- **Trên máy chủ (Local Conda) với 16 luồng song song:**
  ```bash
  python run_pipeline.py --step all --no_docker --workers 16
  ```

- **Trên máy chủ dùng Docker với 16 container song song:**
  ```bash
  python run_pipeline.py --step all --workers 16
  ```

> 💡 **Tốc độ ước tính:** Với 16 workers, thời gian gán nhãn toàn bộ tập dữ liệu rút ngắn từ **~8–10 tiếng xuống còn khoảng 25–40 phút**!

#### Chạy đơn luồng truyền thống:
- Nếu dùng Docker: `python run_pipeline.py --step all`
- Nếu dùng Local Conda: `python run_pipeline.py --step all --no_docker`

### 3. Tùy chọn chạy từng bước riêng biệt:
- Chỉ chuẩn bị dữ liệu: `python run_pipeline.py --step prepare`
- Chỉ chạy tool dán nhãn: `python run_pipeline.py --step label` *(hoặc thêm `--no_docker`)*
- Chỉ tổng hợp nhãn: `python run_pipeline.py --step aggregate`

---

## 📊 Kết quả đầu ra

Sau khi chạy xong, kết quả chính nằm tại:
📂 **`data_work/mimic_cxr_aug_train_labeled.csv`**

File này chứa:
1. **Toàn bộ các cột gốc** của `mimic_cxr_aug_train.csv` (`id`, `image`, `text`, `text_augment`, v.v.).
2. **14 cột bệnh lý CheXpert** theo chuẩn:
   - `No Finding`
   - `Enlarged Cardiomediastinum`
   - `Cardiomegaly`
   - `Lung Lesion`
   - `Lung Opacity`
   - `Edema`
   - `Consolidation`
   - `Pneumonia`
   - `Atelectasis`
   - `Pneumothorax`
   - `Pleural Effusion`
   - `Pleural Other`
   - `Fracture`
   - `Support Devices`
   *(Giá trị nhãn: `1.0` = Dương tính, `0.0` = Âm tính, `-1.0` = Nghi ngờ/Không chắc chắn, Để trống = Không đề cập)*.
3. **Cột `report_labels`**: Chuỗi JSON lưu danh sách đối tượng nhãn chi tiết của từng report bên trong mẫu, cấu trúc:
   ```json
   [
     {"No Finding": null, "Cardiomegaly": 1.0, "Pneumonia": 0.0, ...},
     {"No Finding": 1.0, "Cardiomegaly": null, ...}
   ]
   ```

---

## 🔧 Xử lý sự cố thường gặp (Troubleshooting)

1. **Lỗi `ModuleNotFoundError: No module named 'bllipparser'` hoặc `pandas`:**
   - Đảm bảo bạn đã kích hoạt đúng môi trường conda: `conda activate chexpert`.
   - Dùng lệnh `python -m pip install -r requirements.txt` để ép cài đặt vào đúng Python đang chạy.
2. **Lỗi `LookupError: Resource omw-1.4 not found`:**
   - Chạy lệnh: `python -m nltk.downloader omw-1.4`.
3. **Lỗi `AttributeError: 'DiGraph' object has no attribute 'node'`:**
   - Do `networkx` bị cài phiên bản quá mới (> 2.4). Cần hạ về phiên bản tương thích:
     `python -m pip install networkx==2.3`.
4. **Lỗi `AttributeError: 'BioCAnnotation' object has no attribute 'get_total_location'`:**
   - Do `bioc` bị cài bản 2.0+. Cần ghim đúng bản: `python -m pip install "bioc==1.0" future`.
