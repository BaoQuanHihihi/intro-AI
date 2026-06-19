# Aspect-Based Sentiment Analysis for Vietnamese Tourism Reviews

Hệ thống phân tích cảm xúc đa khía cạnh cho các bài đánh giá du lịch tiếng Việt, sử dụng mô hình ngôn ngữ **PhoBERT** (vinai/phobert-base-v2) kết hợp kiến trúc Multi-Head Classification. Dự án xây dựng và so sánh **hai phương pháp huấn luyện** trên bộ dữ liệu review du lịch:

| Phương pháp | Kiến trúc | F1-Micro | Sentiment Acc | Overall Score |
|---|---|---|---|---|
| **Single Model** | PhoBERT + 6 Linear Heads | **0.8454** | **0.7173** | **0.8070** |
| **5-Fold Ensemble (Proposed)** | PhoBERT + Aspect Heads + Sentiment Heads × 5 Folds | **0.8609** | **0.7558** | **0.8294** |

> **Bài toán:** Cho một bài đánh giá (review) về du lịch, hệ thống cần xác định 6 khía cạnh (`giai_tri`, `luu_tru`, `nha_hang`, `an_uong`, `van_chuyen`, `mua_sam`) và mức độ cảm xúc tương ứng (1–5 sao, 0 nếu không đề cập).

---

## 📁 Cấu trúc thư mục dự án

```text
intro-AI/
├── src/                                # Mã nguồn chính (modular)
│   ├── __init__.py
│   ├── config.py                      #   └─ Cấu hình huấn luyện (TrainConfig, YAML loader)
│   ├── data.py                        #   └─ Dataloader, Dataset, tiền xử lý văn bản
│   ├── metrics.py                     #   └─ Micro-F1, Sentiment Accuracy, Overall Score
│   ├── model.py                       #   └─ JointAspectSentimentModel (PhoBERT + 6 Heads)
│   └── utils.py                       #   └─ Tiện ích (seed, logging, checkpoint)
├── data/                              # Dữ liệu huấn luyện, kiểm thử và script liên quan
│   ├── data-processing-new/           #   └─ Mã nguồn xử lý dữ liệu mới
│   ├── distillation-knowledge/        #   └─ Dữ liệu phục vụ Knowledge Distillation
│   ├── split_datasets/                #   └─ Tập dữ liệu đã chia (K-Fold, train/val/test)
│   ├── targeted_data/                 #   └─ Dữ liệu mục tiêu (chọn lọc theo khía cạnh)
│   ├── train-problem.csv              #   └─ Dữ liệu huấn luyện gốc
│   ├── train_boosted.csv              #   └─ Dữ liệu sau Data Augmentation
│   ├── train_boosted_cleaned.csv      #   └─ Dữ liệu sau Data Augmentation + Làm sạch
│   ├── train_cleaned.csv              #   └─ Dữ liệu gốc đã làm sạch
│   ├── data-cleaner.py                #   └─ Script làm sạch văn bản (teencode, emoji)
│   ├── data-statistic.py              #   └─ Script phân tích & trực quan hóa dữ liệu
│   ├── merge-data.py                  #   └─ Script gộp nhiều file CSV
│   ├── split-data.py                  #   └─ Script chia tập train/test
├── configs/                           # File cấu hình YAML
│   ├── default.yaml                   #   └─ Cấu hình huấn luyện mặc định
│   └── cpu.yaml                       #   └─ Cấu hình cho môi trường CPU
├── models/                            # Thư mục lưu các mô hình đã huấn luyện
│   └── best_model_fold_1..5.pth       #   └─ Checkpoints cho 5 folds
├── predict-and-test/                  # Scripts và kết quả dự đoán, kiểm thử
│   ├── error.txt                      #   └─ Các lỗi dự đoán được ghi nhận
│   ├── find_critical_errors.py        #   └─ Script tìm lỗi nghiêm trọng trong kết quả
│   ├── inference_realtime.py          #   └─ Script dự đoán theo thời gian thực
│   ├── predict_kfold.py               #   └─ Pipeline dự đoán 5-Fold Ensemble
│   └── submission.csv                 #   └─ Kết quả dự đoán cuối cùng cho Kaggle
├── scripts/                           # Script shell hỗ trợ chạy nhanh
│   ├── run_train.sh                   #   └─ Script chạy huấn luyện
│   └── run_predict.sh                 #   └─ Script chạy dự đoán
├── all_code.py                        # Pipeline Single Model (single-file cho Kaggle)
├── all_code_kfold.py                  # Pipeline 5-Fold Ensemble (single-file cho Kaggle)
├── all_code_lora.py                   # Pipeline thử nghiệm LoRA fine-tuning
├── all_code_without_kfold.py          # Pipeline cải tiến không dùng K-Fold
├── all_code_without_kfold_raw.py      # Pipeline baseline thô
├── train.py                           # Script huấn luyện chính (modular)
├── evaluate.py                        # Script đánh giá trên tập validation/test
├── predict.py                         # Script dự đoán → predictions.csv
├── result*.txt                        # Logs và kết quả huấn luyện
├── requirements.txt                   # Danh sách thư viện phụ thuộc
├── .gitignore                         # File cấu hình git ignore
└── README.md                          # Hướng dẫn nhanh này
```

---

## ⚡ Hướng dẫn chạy nhanh (Quickstart)

### 1. Cài đặt môi trường

Yêu cầu **Python ≥ 3.10** và **PyTorch ≥ 2.0** (khuyến nghị cài bản hỗ trợ CUDA tương ứng với card đồ họa).

```bash
python -m pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Dữ liệu đã được bao gồm trong thư mục `data/`. Nếu cần làm sạch lại từ đầu:

```bash
# Làm sạch dữ liệu (xử lý teencode, emoji, ký tự đặc biệt)
cd data
python data-cleaner.py

# Phân tích thống kê dữ liệu và vẽ biểu đồ phân bố
python data-statistic.py
```

### 3. Huấn luyện mô hình

#### A. Huấn luyện Single Model (Modular — chạy local)
```bash
python train.py --config configs/default.yaml
```
*Huấn luyện mô hình PhoBERT + 6 Classification Heads qua 15 epoch, sử dụng Early Stopping (patience=3). Checkpoint tốt nhất lưu tại `outputs/run/best_model.pt`.*

#### B. Huấn luyện 5-Fold Ensemble (Single-file — chạy trên Kaggle)
```bash
python all_code_kfold.py
```
*Huấn luyện 5 mô hình song song theo chiến lược K-Fold Cross Validation. Mỗi fold lưu checkpoint riêng. Đánh giá Ensemble cuối cùng bằng Soft Voting trên tập Test.*

### 4. Đánh giá mô hình

```bash
# Đánh giá trên tập validation (từ checkpoint đã lưu)
python evaluate.py from_model --checkpoint_dir outputs/run

# So sánh file dự đoán với ground truth
python evaluate.py from_files --pred_path predictions.csv --gt_path data/gt_reviews_test.csv
```

### 5. Chạy dự đoán trên tập test

```bash
python predict.py --checkpoint_dir outputs/run --test_path data/gt_reviews_test.csv --output_csv predictions.csv
```
*Kết quả dự đoán được xuất ra file `predictions.csv` theo định dạng: `stt, giai_tri, luu_tru, nha_hang, an_uong, van_chuyen, mua_sam`.*

---

## 📖 Hướng dẫn chi tiết

<details>
<summary><b>Xem chi tiết kiến trúc mô hình và các tham số nâng cao (Click để mở rộng)</b></summary>

### A. Kiến trúc mô hình

#### Single Model (`src/model.py` — `JointAspectSentimentModel`)
*   **Encoder:** PhoBERT (`vinai/phobert-base-v2`) — Mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt.
*   **Pooling:** Lấy vector [CLS] từ `last_hidden_state`.
*   **Heads:** 6 `nn.Linear(hidden_size, 6)` — Mỗi head dự đoán 6 lớp (0: không đề cập, 1–5: mức cảm xúc).
*   **Loss:** Cross-Entropy Loss với Label Smoothing (0.05) và Class Weights (inverse frequency).

#### 5-Fold Ensemble (`all_code_kfold.py` — `SharedEncoderMultiClassModel`)
*   **Encoder:** Tương tự Single Model.
*   **Aspect Heads:** 6 heads (Linear → LayerNorm → ReLU → Dropout → Linear(1)) — Phát hiện khía cạnh (binary).
*   **Sentiment Heads:** 6 heads (Linear → LayerNorm → ReLU → Dropout → Linear(5)) — Phân loại cảm xúc 1–5.
*   **Loss:** Kết hợp `BCEWithLogitsLoss` (aspect, λ=0.7) + `CrossEntropyLoss` (sentiment, λ=0.3).
*   **Ensemble:** Soft Voting — Trung bình xác suất dự đoán của 5 mô hình từ 5 Folds.

### B. Siêu tham số huấn luyện

| Tham số | Single Model | 5-Fold Ensemble |
|---|---|---|
| Encoder | `vinai/phobert-base-v2` | `vinai/phobert-base-v2` |
| Max Length | 256 | 256 |
| Batch Size | 16 | 16 |
| Encoder LR | 2e-5 | 3e-5 |
| Head LR | 2e-5 | 5e-4 |
| Epochs | 15 | 20 |
| Warmup Ratio | 0.06 | 0.1 |
| Early Stopping | 3 epoch | 15 epoch |
| Optimizer | AdamW | AdamW |
| Scheduler | Linear Warmup + Decay | Cosine Warmup |
| Grad Accumulation | 1 | 2 |

### C. Tiền xử lý dữ liệu (`data/data-cleaner.py`)
*   **Chuẩn hóa teencode:** `ks` → `khách sạn`, `ko` → `không`, `nv` → `nhân viên`, ...
*   **Chuyển đổi emoji:** `😊` → `hài lòng`, `😡` → `tồi tệ tức giận`, ...
*   **Làm sạch:** Xóa dấu câu thừa, chuẩn hóa khoảng trắng, lowercase.

### D. Các chỉ số đánh giá (`src/metrics.py`)
*   **Micro-F1 Aspect Presence:** Binary F1 — Đánh giá khả năng phát hiện khía cạnh (nhãn > 0 vs = 0).
*   **Sentiment Accuracy:** Accuracy trên các mẫu có nhãn > 0 — Đánh giá phân loại cảm xúc 1–5.
*   **Overall Score:** `0.7 × Micro-F1 + 0.3 × Sentiment Accuracy`.

</details>

---

## 📊 Kết quả thực nghiệm

### Kết quả 5-Fold Cross Validation

| Fold | Train Loss | F1-Micro | Sentiment Acc | Overall Score |
| :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 0.0191 | 0.8527 | 0.6875 | 0.8031 |
| Fold 2 | 0.0678 | 0.8550 | 0.6753 | 0.8011 |
| Fold 3 | 0.0189 | 0.8628 | 0.6905 | 0.8111 |
| Fold 4 | 0.0305 | 0.8655 | 0.6760 | 0.8086 |
| Fold 5 | 0.0169 | 0.8377 | 0.6717 | 0.7879 |
| **Mean** | - | - | - | **0.8024** |

### Kết quả đánh giá trên tập Test độc lập

| Mô hình | F1-Micro | Sentiment Acc | Overall Score |
|---|---|---|---|
| Single Model | 0.8547 | 0.7072 | 0.8104 |
| **5-Fold Ensemble** | **0.8628** | **0.7497** | **0.8289** |

---

## 👥 Phân công nhiệm vụ thành viên (Task Allocation)

Dưới đây là bảng phân công công việc chi tiết cho các thành viên trong nhóm thực hiện dự án:

| STT | Họ và tên | MSSV | Nhiệm vụ cụ thể | Đóng góp |
| :---: | :--- | :---: | :--- | :---: |
| 1 | **Đặng Bảo Quân** | 202416319 | - Thu thập, làm sạch và tăng cường dữ liệu (`data-cleaner.py`, `merge-data.py`, `split-data.py`).<br>- Phân tích thống kê và trực quan hóa phân bố dữ liệu (`data-statistic.py`).<br>- Xây dựng pipeline tiền xử lý văn bản (xử lý teencode, emoji, chuẩn hóa).<br>- Biên soạn báo cáo kỹ thuật tổng hợp. | 100% |
| 2 | **Nguyễn Hoàng Gia** | 202400040 | - Thiết kế kiến trúc mô hình `JointAspectSentimentModel` và `SharedEncoderMultiClassModel` (`src/model.py`, `all_code_kfold.py`).<br>- Xây dựng hệ thống cấu hình YAML và pipeline huấn luyện (`src/config.py`, `train.py`).<br>- Triển khai chiến lược 5-Fold Cross Validation và Ensemble Soft Voting.<br>- Tinh chỉnh siêu tham số và tối ưu hóa hiệu năng mô hình. | 100% |
| 3 | **Nguyễn Tuấn Long** | 202416269 | - Xây dựng module đánh giá và các chỉ số Micro-F1, Sentiment Accuracy, Overall Score (`src/metrics.py`, `evaluate.py`).<br>- Triển khai pipeline dự đoán trên tập test (`predict.py`).<br>- Xây dựng Dataset và DataLoader cho bài toán multi-head classification (`src/data.py`).<br>- Chuẩn bị tài liệu thuyết trình (slides) và thực hiện kiểm thử hệ thống. | 100% |

---

## 📄 Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.10+ |
| Deep Learning Framework | PyTorch ≥ 2.0 |
| Pretrained Model | [PhoBERT-base-v2](https://huggingface.co/vinai/phobert-base-v2) (VinAI) |
| Tokenizer | HuggingFace Transformers (`AutoTokenizer`) |
| Optimizer | AdamW (weight decay, differential LR) |
| Scheduler | Linear / Cosine Warmup |
| Đánh giá | scikit-learn style metrics (custom) |
| Môi trường chạy | Kaggle Notebooks (GPU T4/P100), Local CUDA |
