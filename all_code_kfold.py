#!/usr/bin/env python3
"""
Single-file script for Kaggle: 5-Fold Shared PhoBERT Encoder with 6 Independent Multi-Class Sentiment Heads.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import KFold

# ==========================================
# 1. CONSTANTS & METRICS
# ==========================================
ASPECT_COLS = ("giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam")
TEXT_CANDIDATES = ("review", "Review", "text", "content", "comment", "noi_dung", "noidung")

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
# 2. UTILS
# ==========================================
def setup_logging():
    log = logging.getLogger("review_analytics")
    if log.handlers: return log
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    return log

LOG = setup_logging()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def try_cuda_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 3. CONFIGURATION
# ==========================================
@dataclass
class TrainConfig:
    model_name: str = "vinai/phobert-base-v2"
    max_length: int = 256
    batch_size: int = 16
    grad_accum_steps: int = 2
    
    encoder_lr: float = 3e-5
    head_lr: float = 5e-4
    
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    seed: int = 42
    n_folds: int = 5
    early_stopping_patience: int = 15
    max_grad_norm: float = 1.0
    
    train_path: str = "/kaggle/input/datasets/baoquanhihihi/dataset-for-introduction-to-ai-course-hust/train_split.csv"
    test_path: str = "/kaggle/input/datasets/baoquanhihihi/dataset-for-introduction-to-ai-course-hust/test_split.csv" 
    output_dir: str = "/kaggle/working/outputs_kfold"
    
    lambda_aspect: float = 0.7 
    lambda_sentiment: float = 0.3

# ==========================================
# 4. DATA PROCESSING
# ==========================================
def find_text_column(df):
    for cand in TEXT_CANDIDATES:
        if cand in df.columns: return cand
    raise ValueError("Không thấy cột text")

def load_and_clean_dataset(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path, encoding="utf-8")
    text_col = find_text_column(df)
    df["_text"] = df[text_col].astype(str).str.replace("\r\n", "\n").str.strip()
    df["_text"] = df["_text"].replace("", " ")
    return df

class ReviewAspectDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, label_matrix=None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = label_matrix

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        item = {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.labels is not None: item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ==========================================
# 5. MODEL ARCHITECTURE
# ==========================================
class SharedEncoderMultiClassModel(nn.Module):
    def __init__(self, model_name: str, num_aspects: int = 6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        
        self.aspect_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ) for _ in range(num_aspects)
        ])

        self.sentiment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 5)
            ) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  
        aspect_logits = torch.stack([h(pooled).squeeze(-1) for h in self.aspect_heads], dim=1)
        sentiment_logits = torch.stack([h(pooled) for h in self.sentiment_heads], dim=1)
        return aspect_logits, sentiment_logits

def build_model_and_tokenizer(model_name: str):
    model = SharedEncoderMultiClassModel(model_name)
    return model

# ==========================================
# 6. EVALUATION LOGIC
# ==========================================
@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_true, all_aspect_preds, all_sentiment_preds = [], [], []
    for batch in loader:
        aspect_logits, sentiment_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        aspect_preds = (torch.sigmoid(aspect_logits) >= 0.5).long()
        sentiment_preds = torch.argmax(sentiment_logits, dim=-1) + 1
        
        all_true.append(labels_to_numpy(batch["labels"]))
        all_aspect_preds.append(labels_to_numpy(aspect_preds))
        all_sentiment_preds.append(labels_to_numpy(sentiment_preds))
        
    return compute_all_metrics(
        np.concatenate(all_true, axis=0), 
        np.concatenate(all_aspect_preds, axis=0), 
        np.concatenate(all_sentiment_preds, axis=0)
    )

# ==========================================
# 7. MAIN 5-FOLD TRAINING PIPELINE
# ==========================================
def train() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = try_cuda_device()
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    LOG.info(f"🚀 Bắt đầu huấn luyện hệ thống {cfg.n_folds}-Fold Cross Validation.")
    LOG.info(f"Thiết bị phần cứng: {device}")

    full_df = load_and_clean_dataset(Path(cfg.train_path))
    test_df = load_and_clean_dataset(Path(cfg.test_path))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    test_ds = ReviewAspectDataset(
        test_df["_text"].tolist(), 
        tokenizer, 
        cfg.max_length, 
        test_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)
    
    # Phân chia dữ liệu theo K-Fold
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    
    # Mảng lưu giữ thành tích tối ưu của các fold nhằm tính điểm phân phối cuối cùng
    fold_best_scores = []

    # VÒNG LẶP CHÍNH KHỞI TẠO TỪNG FOLD
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_df), 1):
        LOG.info(f"\n=================== 📦 FOLD {fold}/{cfg.n_folds} ===================")
        
        # Phân mảnh dataset tương ứng với index của Fold hiện tại
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        
        # Tạo mới Model cho riêng fold này
        model = build_model_and_tokenizer(cfg.model_name)
        model.to(device)
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        actual_model = model.module if isinstance(model, nn.DataParallel) else model

        # Đóng gói dữ liệu vào Dataset & DataLoader
        train_ds = ReviewAspectDataset(
            train_df["_text"].tolist(), 
            tokenizer, 
            cfg.max_length, 
            train_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
        )
        val_ds = ReviewAspectDataset(
            val_df["_text"].tolist(), 
            tokenizer, 
            cfg.max_length, 
            val_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
        )
        
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

        # Tính toán lại trọng số lớp cân bằng riêng cho dữ liệu của từng Fold huấn luyện
        y_train = train_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
        pos_counts = (y_train > 0).sum(axis=0)
        neg_counts = (y_train == 0).sum(axis=0)
        pos_weights = torch.tensor(neg_counts / np.clip(pos_counts, 1, None), dtype=torch.float).to(device)
        
        criterion_aspect = [nn.BCEWithLogitsLoss(pos_weight=pos_weights[a]) for a in range(6)]
        criterion_sentiment = nn.CrossEntropyLoss()

        # Thiết lập các nhóm tối ưu hóa siêu tham số độc lập
        encoder_params = list(actual_model.encoder.named_parameters())
        head_params = list(actual_model.aspect_heads.named_parameters()) + list(actual_model.sentiment_heads.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        
        optim_groups = [
            {"params": [p for n, p in encoder_params if not any(nd in n for nd in no_decay)], "lr": cfg.encoder_lr, "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in encoder_params if any(nd in n for nd in no_decay)], "lr": cfg.encoder_lr, "weight_decay": 0.0},
            {"params": [p for n, p in head_params if not any(nd in n for nd in no_decay)], "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in head_params if any(nd in n for nd in no_decay)], "lr": cfg.head_lr, "weight_decay": 0.0},
        ]

        optim = AdamW(optim_groups, eps=1e-8)
        total_steps = math.ceil(len(train_loader) / cfg.grad_accum_steps) * cfg.num_epochs
        sched = get_cosine_schedule_with_warmup(
            optimizer=optim, num_warmup_steps=int(total_steps * cfg.warmup_ratio), num_training_steps=total_steps
        )
        
        # Biến theo dõi trạng thái tối ưu của Fold hiện tại
        best_fold_score = -1.0
        patience = cfg.early_stopping_patience

        # VÒNG LẶP EPOCH CỦA FOLD
        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Fold {fold} | Ep {epoch}/{cfg.num_epochs}", file=sys.stdout, leave=True)

            for step, batch in enumerate(pbar):
                labels = batch["labels"].to(device)
                aspect_logits, sentiment_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                
                aspect_targets = (labels > 0).float()
                loss_aspect = 0.0
                loss_sentiment = 0.0
                valid_sentiment_heads = 0 
                
                for a in range(6):
                    loss_aspect += criterion_aspect[a](aspect_logits[:, a], aspect_targets[:, a])
                    valid_mask = labels[:, a] > 0 
                    if valid_mask.any():
                        preds_valid = sentiment_logits[valid_mask, a]
                        targets_valid = (labels[valid_mask, a] - 1).long() 
                        loss_sentiment += criterion_sentiment(preds_valid, targets_valid)
                        valid_sentiment_heads += 1
                
                loss_aspect = loss_aspect / 6.0
                loss_sentiment = loss_sentiment / max(1, valid_sentiment_heads) 
                loss = (cfg.lambda_aspect * loss_aspect) + (cfg.lambda_sentiment * loss_sentiment)
                
                (loss / cfg.grad_accum_steps).backward()

                if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)

                epoch_loss += float(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Đánh giá cuối mỗi epoch trên tập Validation cụ thể của fold này
            metrics = evaluate_epoch(model, val_loader, device)
            LOG.info(
                f"--> [Fold {fold} - Ep {epoch}] Train Loss: {epoch_loss/len(train_loader):.4f} | "
                f"F1-Micro: {metrics['micro_f1']:.4f} | "
                f"Sentiment Acc: {metrics['sentiment_accuracy']:.4f} | "
                f"OVERALL: {metrics['overall_score']:.4f}"
            )

            # Cơ chế lưu trữ checkpoint định danh riêng cho từng fold đơn lẻ
            if metrics["overall_score"] > best_fold_score:
                best_fold_score = metrics["overall_score"]
                patience = cfg.early_stopping_patience
                LOG.info(f"   [!] Đạt điểm mới cho Fold {fold}: {best_fold_score:.4f}")
                
                save_path = os.path.join(cfg.output_dir, f"best_model_fold_{fold}.pth")
                torch.save(actual_model.state_dict(), save_path)
                LOG.info(f"   [+] Đã bảo lưu trọng số tối ưu vào: {save_path}")
            else:
                patience -= 1
                if patience <= 0:
                    LOG.info(f"Dừng sớm Fold {fold} tại Epoch {epoch} do chỉ số không cải thiện thêm.")
                    break
        
        # Ghi nhận kết quả tối ưu mà fold này đạt được
        fold_best_scores.append(best_fold_score)
        LOG.info(f"🎯 Hoàn thành Fold {fold}. Điểm Overall cao nhất đạt được: {best_fold_score:.4f}\n")

        # Giải phóng bộ nhớ đệm GPU tránh tràn VRAM khi khởi chạy Fold tiếp theo
        del model, optim, sched, train_loader, val_loader
        torch.cuda.empty_cache()


    # ==========================================
    # 8. FINAL SYSTEM VALIDATION REPORT
    # ==========================================
    LOG.info("\n=================== 📊 BÁO CÁO HIỆU SUẤT TỔNG THỂ K-FOLD ===================")
    for i, score in enumerate(fold_best_scores, 1):
        LOG.info(f"Fold {i}: {score:.4f}")
    LOG.info(f"➔ Độ chính xác trung bình ({cfg.n_folds}-Fold Mean Score): {np.mean(fold_best_scores):.4f}")



    # ==========================================
    # 9. EVALUATE ENSEMBLE ON HOLD-OUT TEST SET
    # ==========================================
    LOG.info("\n=================== 🏆 ĐÁNH GIÁ ENSEMBLE TRÊN TẬP TEST ĐỘC LẬP ===================")
    
    # Thu thập tất cả trọng số của 5 Folds đã lưu
    model_paths = [os.path.join(cfg.output_dir, f"best_model_fold_{i}.pth") for i in range(1, cfg.n_folds + 1)]
    models = []
    
    # Load lại cấu trúc và nạp trọng số của từng fold vào danh sách
    for pth in model_paths:
        m = SharedEncoderMultiClassModel(cfg.model_name)
        m.load_state_dict(torch.load(pth, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)
        
    all_true = []
    ensemble_aspect_probs = []
    ensemble_sentiment_probs = []

    # Dự đoán
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            batch_aspect_probs = 0.0
            batch_sentiment_probs = 0.0
            
            # Tính trung bình xác suất của 5 model
            for m in models:
                aspect_logits, sentiment_logits = m(input_ids, attention_mask)
                
                batch_aspect_probs += torch.sigmoid(aspect_logits)
                batch_sentiment_probs += torch.softmax(sentiment_logits, dim=-1)
                
            batch_aspect_probs /= len(models)
            batch_sentiment_probs /= len(models)
            
            all_true.append(labels_to_numpy(batch["labels"]))
            ensemble_aspect_probs.append(labels_to_numpy(batch_aspect_probs))
            ensemble_sentiment_probs.append(labels_to_numpy(batch_sentiment_probs))

    # Gộp kết quả dự đoán của toàn bộ tập Test
    y_true_test = np.concatenate(all_true, axis=0)
    final_aspect_preds = (np.concatenate(ensemble_aspect_probs, axis=0) >= 0.5).astype(np.int64)
    final_sentiment_preds = np.argmax(np.concatenate(ensemble_sentiment_probs, axis=0), axis=-1) + 1

    # Tính toán chỉ số hiệu năng cuối cùng của hệ thống trên tập Test thực tế
    test_metrics = compute_all_metrics(y_true_test, final_aspect_preds, final_sentiment_preds)
    
    LOG.info(f"➔ [TEST SET] F1-Micro Aspect: {test_metrics['micro_f1']:.4f}")
    LOG.info(f"➔ [TEST SET] Sentiment Accuracy: {test_metrics['sentiment_accuracy']:.4f}")
    LOG.info(f"➔ 🌟 [TEST SET] ENSEMBLE OVERALL SCORE: {test_metrics['overall_score']:.4f}")

if __name__ == "__main__":
    train()