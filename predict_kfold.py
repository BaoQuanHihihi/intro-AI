#!/usr/bin/env python3
"""
Independent Inference Script for Kaggle: Predict using chosen folds from 5-Fold Trained Models.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. CONSTANTS & CONFIG COPIED FROM TRAIN
# ==========================================
ASPECT_COLS = ("giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam")
TEXT_CANDIDATES = ("review", "Review", "text", "content", "comment", "noi_dung", "noidung")

class TrainConfig:
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 256
    batch_size: int = 16
    output_dir: str = "models"
    test_path: str = "data/split_datasets/test_split.csv"

# ==========================================
# 2. DATASET & MODEL DEFINITIONS
# ==========================================
def find_text_column(df):
    for cand in TEXT_CANDIDATES:
        if cand in df.columns: return cand
    raise ValueError("Không thấy cột text trong file test_split.csv")

class ReviewAspectDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}

class SharedEncoderMultiClassModel(nn.Module):
    def __init__(self, model_name: str, num_aspects: int = 6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        
        self.aspect_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)
            ) for _ in range(num_aspects)
        ])

        self.sentiment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5)
            ) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  
        aspect_logits = torch.stack([h(pooled).squeeze(-1) for h in self.aspect_heads], dim=1)
        sentiment_logits = torch.stack([h(pooled) for h in self.sentiment_heads], dim=1)
        return aspect_logits, sentiment_logits

def labels_to_numpy(y: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    return np.asarray(y, dtype=np.int64)

def micro_f1_aspect_presence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = (y_true > 0).astype(np.int64).ravel()
    yp = (y_pred > 0).astype(np.int64).ravel()
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    
    if tp == 0 and (fp + fn) == 0: return 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0: return 0.0
    return 2.0 * prec * rec / (prec + rec)

def sentiment_accuracy_masked(y_true: np.ndarray, y_pred_sentiment: np.ndarray) -> float:
    mask = y_true > 0
    if not mask.any(): return 1.0
    correct = (y_true[mask] == y_pred_sentiment[mask]).astype(np.float64)
    return float(correct.mean())

def compute_all_metrics(y_true: np.ndarray, aspect_preds: np.ndarray, sentiment_preds: np.ndarray) -> dict[str, float]:
    mf1 = micro_f1_aspect_presence(y_true, aspect_preds)
    sa = sentiment_accuracy_masked(y_true, sentiment_preds)
    return {
        "micro_f1": mf1,
        "sentiment_accuracy": sa,
        "overall_score": 0.7 * mf1 + 0.3 * sa,
    }

# ==========================================
# 3. INFERENCE PIPELINE
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Inference Script using specific models from K-Fold.")
    parser.add_argument(
        "--folds", 
        type=int, 
        nargs="+", 
        default=[1,2,3,4,5],
        help="Danh sách các fold muốn sử dụng để predict (Ví dụ: --folds 1 2 5)"
    )
    parser.add_argument("--test_path", type=str, default=TrainConfig.test_path, help="Đường dẫn tới file test csv")
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir, help="Thư mục chứa các file .pth đã train")
    parser.add_argument("--save_csv", type=str, default="submission.csv", help="Tên file csv kết quả đầu ra")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Selected Folds for Ensemble: {args.folds}")

    # Load dữ liệu test
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Không tìm thấy file test tại: {args.test_path}")
        
    df_test = pd.read_csv(args.test_path, encoding="utf-8")
    text_col = find_text_column(df_test)
    
    # Tiền xử lý text cơ bản giống file train
    texts = df_test[text_col].astype(str).str.replace("\r\n", "\n").str.strip().replace("", " ").tolist()

    # Khởi tạo Tokenizer & Dataloader
    tokenizer = AutoTokenizer.from_pretrained(TrainConfig.model_name)
    test_ds = ReviewAspectDataset(texts, tokenizer, TrainConfig.max_length)
    test_loader = DataLoader(test_ds, batch_size=TrainConfig.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Khởi tạo và nạp trọng số cho các Model được chọn
    models = []
    for f in args.folds:
        model_path = os.path.join(args.output_dir, f"best_model_fold_{f}.pth")
        if not os.path.exists(model_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy checkpoint cho Fold {f} tại {model_path}. Bỏ qua fold này.")
            continue
            
        print(f"📦 Đang tải trọng số Fold {f} từ: {model_path}")
        model = SharedEncoderMultiClassModel(TrainConfig.model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    if not models:
        print("❌ Lỗi: Không có model fold nào được nạp thành công! Kết thúc script.")
        return

    # Tiến hành Predict (Ensemble dựa trên tính trung bình xác suất)
    ensemble_aspect_probs = []
    ensemble_sentiment_probs = []

    print("🚀 Bắt đầu quá trình dự đoán trên tập test...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            batch_aspect_probs = 0.0
            batch_sentiment_probs = 0.0
            
            for model in models:
                aspect_logits, sentiment_logits = model(input_ids, attention_mask)
                
                # Tính Sigmoid cho khía cạnh và Softmax cho mức độ cảm xúc
                batch_aspect_probs += torch.sigmoid(aspect_logits)
                batch_sentiment_probs += torch.softmax(sentiment_logits, dim=-1)
                
            # Chia trung bình cho số lượng model tham gia ensemble
            batch_aspect_probs /= len(models)
            batch_sentiment_probs /= len(models)
            
            ensemble_aspect_probs.append(batch_aspect_probs.cpu().numpy())
            ensemble_sentiment_probs.append(batch_sentiment_probs.cpu().numpy())

    # Gộp mảng từ các batch
    final_aspect_probs = np.concatenate(ensemble_aspect_probs, axis=0)      # Shape: (N, 6)
    final_sentiment_probs = np.concatenate(ensemble_sentiment_probs, axis=0) # Shape: (N, 6, 5)

    aspect_preds = (final_aspect_probs >= 0.5).astype(np.int64) 
    sentiment_preds = np.argmax(final_sentiment_probs, axis=-1) + 1 
    final_preds = np.where(aspect_preds > 0, sentiment_preds, 0)

    # # Ghi kết quả vào DataFrame
    # submission_df = pd.DataFrame(final_preds, columns=ASPECT_COLS)
    
    # # Nếu file gốc có cột ID hoặc muốn giữ lại cột Text ban đầu để đối chiếu
    # if "id" in df_test.columns:
    #     submission_df.insert(0, "id", df_test["id"])
    # elif "ID" in df_test.columns:
    #     submission_df.insert(0, "ID", df_test["ID"])

    # # Ghi kết quả vào DataFrame
    # submission_df = pd.DataFrame(final_preds, columns=ASPECT_COLS)
    # print(f"🎉 Dự đoán hoàn tất! File kết quả đã được lưu tại: {args.save_csv}")
    # print(submission_df.head())

    # --- BỔ SUNG: Kiểm tra và tính toán Metric nếu tập test có sẵn nhãn ground truth ---
    if all(col in df_test.columns for col in ASPECT_COLS):
        print("\n=================== 📊 ĐÁNH GIÁ CHẤT LƯỢNG ENSEMBLE TRÊN TẬP TEST ===================")
        y_true_test = df_test[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
        
        # Tính toán chỉ số dựa trên kết quả đã argmax / threshold 0.5
        test_metrics = compute_all_metrics(y_true_test, aspect_preds, sentiment_preds)
        
        print(f"➔ [TEST SET] F1-Micro Aspect: {test_metrics['micro_f1']:.4f}")
        print(f"➔            Sentiment Accuracy: {test_metrics['sentiment_accuracy']:.4f}")
        print(f"➔            ENSEMBLE OVERALL SCORE (0.7*F1 + 0.3*Acc): {test_metrics['overall_score']:.4f}")
        print("=================================================================================\n")
    else:
        print("\n⚠️ Không tìm thấy đầy đủ các cột nhãn trong file test_split.csv. Bỏ qua bước tính điểm Metric.")

if __name__ == "__main__":
    main()