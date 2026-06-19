#!/usr/bin/env python3
"""
Independent Inference Script for Kaggle: Predict using chosen folds from 5-Fold Trained Models.
Enhanced version: Computes Single Folds, K-Fold Mean, Ensemble Progression, and Conditional Confusion Matrix.
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
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. CONSTANTS & CONFIG COPIED FROM TRAIN
# ==========================================
ASPECT_COLS = ("giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam")
TEXT_CANDIDATES = ("review", "Review", "text", "content", "comment", "noi_dung", "noidung")

class TrainConfig:
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 256
    batch_size: int = 16
    output_dir: str = "../models"
    test_path: str = "../data/split_datasets/test_split.csv"

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
    parser = argparse.ArgumentParser(description="Inference Script with Automated Report Generator.")
    parser.add_argument("--folds", type=int, nargs="+", default=[1,2,3,4,5], help="Danh sách các fold huấn luyện")
    parser.add_argument("--test_path", type=str, default=TrainConfig.test_path, help="Đường dẫn tới file test csv")
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir, help="Thư mục chứa file .pth")
    parser.add_argument("--save_csv", type=str, default="submission.csv", help="File kết quả")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Không tìm thấy file test tại: {args.test_path}")
        
    df_test = pd.read_csv(args.test_path, encoding="utf-8")
    text_col = find_text_column(df_test)
    texts = df_test[text_col].astype(str).str.replace("\r\n", "\n").str.strip().replace("", " ").tolist()

    tokenizer = AutoTokenizer.from_pretrained(TrainConfig.model_name)
    test_ds = ReviewAspectDataset(texts, tokenizer, TrainConfig.max_length)
    test_loader = DataLoader(test_ds, batch_size=TrainConfig.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    has_ground_truth = all(col in df_test.columns for col in ASPECT_COLS)
    y_true_test = df_test[list(ASPECT_COLS)].to_numpy(dtype=np.int64) if has_ground_truth else None


    fold_metrics = {}
    models = []
    
    sum_aspect_probs = None
    sum_sentiment_probs = None
    valid_fold_count = 0

    print("🔄 Bắt đầu nạp mô hình và đánh giá cục bộ từng Fold...")
    for f in args.folds:
        model_path = os.path.join(args.output_dir, f"best_model_fold_{f}.pth")
        if not os.path.exists(model_path):
            print(f"⚠️ Checkpoint Fold {f} không tìm thấy tại {model_path}. Bỏ qua.")
            continue
            
        model = SharedEncoderMultiClassModel(TrainConfig.model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        valid_fold_count += 1

        # Thực hiện dự đoán đơn lẻ của Fold hiện tại trên tập test
        fold_aspect_probs = []
        fold_sentiment_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                aspect_logits, sentiment_logits = model(input_ids, attention_mask)
                
                fold_aspect_probs.append(torch.sigmoid(aspect_logits).cpu().numpy())
                fold_sentiment_probs.append(torch.softmax(sentiment_logits, dim=-1).cpu().numpy())
        
        f_asp_p = np.concatenate(fold_aspect_probs, axis=0)
        f_sen_p = np.concatenate(fold_sentiment_probs, axis=0)
        
        # Khởi tạo ma trận tích lũy Ensemble
        if sum_aspect_probs is None:
            sum_aspect_probs = np.zeros_like(f_asp_p)
            sum_sentiment_probs = np.zeros_like(f_sen_p)
            
        sum_aspect_probs += f_asp_p
        sum_sentiment_probs += f_sen_p

        # Tính toán metric độc lập cho fold này nếu có Ground truth
        if has_ground_truth:
            f_asp_pred = (f_asp_p >= 0.5).astype(np.int64)
            f_sen_pred = np.argmax(f_sen_p, axis=-1) + 1
            fold_metrics[f] = compute_all_metrics(y_true_test, f_asp_pred, f_sen_pred)
            print(f"✅ Fold {f} Loaded | Test Score - Overall: {fold_metrics[f]['overall_score']:.4f}")

    if valid_fold_count == 0:
        print("❌ Lỗi: Không có checkpoint nào được nạp thành công.")
        return

    # Tính toán kết quả Ensemble (Chia trung bình xác suất)
    final_aspect_probs = sum_aspect_probs / valid_fold_count
    final_sentiment_probs = sum_sentiment_probs / valid_fold_count

    aspect_preds = (final_aspect_probs >= 0.5).astype(np.int64) 
    sentiment_preds = np.argmax(final_sentiment_probs, axis=-1) + 1 
    final_preds = np.where(aspect_preds > 0, sentiment_preds, 0)

    # Xuất file submission
    submission_df = pd.DataFrame(final_preds, columns=ASPECT_COLS)
    if "id" in df_test.columns: submission_df.insert(0, "id", df_test["id"])
    elif "ID" in df_test.columns: submission_df.insert(0, "ID", df_test["ID"])
    submission_df.to_csv(args.save_csv, index=False)
    print(f"\n🎉 Xuất file kết quả thành công tại: {args.save_csv}")

    # ==========================================
    # LOGIC IN DỮ LIỆU BÁO CÁO (AUTOMATED REPORTS)
    # ==========================================
    if not has_ground_truth:
        print("\n⚠️ Không có Ground Truth trên tập test để tính toán số liệu cho các bảng.")
        return

    # Tính metric cuối của Ensemble Model
    ensemble_metrics = compute_all_metrics(y_true_test, aspect_preds, sentiment_preds)

    # --- BẢNG 1: THỐNG KÊ HIỆU NĂNG 5-FOLD ---
    print("\n👉 BẢNG 1: Dữ liệu điền vào 'Bảng thống kê hiệu năng tối ưu trên không gian 5-Fold'")
    print("-" * 75)
    f1_list, acc_list, overall_list = [], [], []
    for f in sorted(fold_metrics.keys()):
        m = fold_metrics[f]
        print(f"Fold {f}  ->  Aspect: {m['micro_f1']:.4f} | Sentiment: {m['sentiment_accuracy']:.4f} | Overall: {m['overall_score']:.4f}")
        f1_list.append(m['micro_f1'])
        acc_list.append(m['sentiment_accuracy'])
        overall_list.append(m['overall_score'])
    
    mean_f1, mean_acc, mean_overall = np.mean(f1_list), np.mean(acc_list), np.mean(overall_list)
    print("-" * 75)
    print(f"Giá trị trung bình (mu) -> Aspect: {mean_f1:.4f} | Sentiment: {mean_acc:.4f} | Overall: {mean_overall:.4f}")

    # --- BẢNG 2: SO SÁNH ĐỐI CHỨNG ENSEMBLE ---
    # Lấy đại diện Fold 1 làm Single Model (hoặc fold đầu tiên tìm thấy)
    rep_fold = sorted(fold_metrics.keys())[0]
    single_m = fold_metrics[rep_fold]
    
    print("\n👉 BẢNG 2: Dữ liệu điền vào 'Bảng so sánh đối chứng hiệu năng Single vs Ensemble'")
    print("-" * 75)
    print(f"Single Fold Model (Fold {rep_fold}) -> Aspect: {single_m['micro_f1']:.4f} | Sentiment: {single_m['sentiment_accuracy']:.4f} | Overall: {single_m['overall_score']:.4f}")
    print(f"5-Fold Ensemble Model         -> Aspect: {ensemble_metrics['micro_f1']:.4f} | Sentiment: {ensemble_metrics['sentiment_accuracy']:.4f} | Overall: {ensemble_metrics['overall_score']:.4f}")
    print("-" * 75)
    print(f"Biên độ tăng trưởng (Delta)   -> Aspect: {+(ensemble_metrics['micro_f1'] - single_m['micro_f1']):+.4f} | Sentiment: {+(ensemble_metrics['sentiment_accuracy'] - single_m['sentiment_accuracy']):+.4f} | Overall: {+(ensemble_metrics['overall_score'] - single_m['overall_score']):+.4f}")

    # --- BẢNG 3: MA TRẬN NHẦM LẪN SỐ SAO (CONDITIONAL MASKED) ---
    print("\n👉 BẢNG 3: Dữ liệu điền vào 'Ma trận nhầm lẫn (%) tác vụ Sentiment (1-5 sao)'")
    print("-" * 75)
    
    # Tạo mặt nạ: chỉ tính những vị trí khía cạnh thực sự tồn tại (Ground Truth > 0)
    mask = y_true_test > 0
    y_true_filtered = y_true_test[mask]
    y_pred_filtered = sentiment_preds[mask]

    # Tính confusion matrix thô (miền giá trị từ 1 đến 5)
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[1, 2, 3, 4, 5])
    
    # Chuẩn hóa về tỷ lệ phần trăm (%) theo từng hàng thực tế
    cm_percent = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

    # In ma trận nhầm lẫn trực quan để điền vào LaTeX
    print("Thực tế \\ Dự đoán |   1 Sao   |   2 Sao   |   3 Sao   |   4 Sao   |   5 Sao   |")
    for i, row in enumerate(cm_percent, 1):
        row_str = f"     {i} Sao       |"
        for val in row:
            row_str += f"  {val:6.2f}% |"
        print(row_str)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()