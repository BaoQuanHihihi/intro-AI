#!/usr/bin/env python3
"""
Single-file script for Kaggle: Dual-Head Aspect-Sentiment Model.
Đã tích hợp: Mean Pooling (GAP) & Progressive Unfreezing cho PhoBERT.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

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
# 3. CONFIGURATION (Đã tối ưu)
# ==========================================
@dataclass
class TrainConfig:
    model_name: str = "vinai/phobert-base-v2"
    fallback_model_name: str = "xlm-roberta-base"
    max_length: int = 256
    
    batch_size: int = 64
    grad_accum_steps: int = 2
    
    encoder_lr: float = 3e-5
    head_lr: float = 5e-4
    
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    seed: int = 42
    val_ratio: float = 0.1
    early_stopping_patience: int = 20
    max_grad_norm: float = 1.0
    num_workers: int = 2
    
    train_path: str = "/kaggle/input/datasets/baoquanhihihi/dataset-for-introduction-to-ai-course-hust/train_boosted_cleaned.csv"
    output_dir: str = "/kaggle/working/outputs_new"
    
    lambda_aspect: float = 0.7 
    lambda_sentiment: float = 0.3

# ==========================================
# 4. DATA PROCESSING
# ==========================================
def find_text_column(df):
    for cand in TEXT_CANDIDATES:
        if cand in df.columns: return cand
    raise ValueError("Không thấy cột text")

def load_train_val_frames(train_path: Path, val_ratio: float, seed: int):
    df = pd.read_csv(train_path, encoding="utf-8")
    text_col = find_text_column(df)
    df["_text"] = df[text_col].astype(str).str.replace("\r\n", "\n").str.strip()
    df["_text"] = df["_text"].replace("", " ")
    
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_val = max(1, int(round(len(df) * val_ratio)))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]
    return df.iloc[tr_idx].copy().reset_index(drop=True), df.iloc[va_idx].copy().reset_index(drop=True)

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
# 5. DUAL-HEAD HIERARCHICAL MODEL (Với Mean Pooling)
# ==========================================
class DualHeadAspectSentimentModel(nn.Module):
    def __init__(self, model_name: str, num_aspects: int = 6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        
        self.aspect_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)) 
            for _ in range(num_aspects)
        ])

        self.sentiment_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1)) 
            for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        
        aspect_logits = torch.stack([h(pooled).squeeze(-1) for h in self.aspect_heads], dim=1)
        sentiment_logits = torch.stack([h(pooled).squeeze(-1) for h in self.sentiment_heads], dim=1)
        return aspect_logits, sentiment_logits

def build_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DualHeadAspectSentimentModel(model_name)
    return model, tokenizer

# ==========================================
# 6. TRAINING LOGIC & PROGRESSIVE UNFREEZING
# ==========================================
@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_true, all_aspect_preds, all_sentiment_preds = [], [], []
    for batch in loader:
        aspect_logits, sentiment_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        
        aspect_preds = (torch.sigmoid(aspect_logits) >= 0.5).long()
        sentiment_preds = torch.round(sentiment_logits).clamp(1, 5).long()
        
        all_true.append(labels_to_numpy(batch["labels"]))
        all_aspect_preds.append(labels_to_numpy(aspect_preds))
        all_sentiment_preds.append(labels_to_numpy(sentiment_preds))
        
    return compute_all_metrics(
        np.concatenate(all_true, axis=0), 
        np.concatenate(all_aspect_preds, axis=0), 
        np.concatenate(all_sentiment_preds, axis=0)
    )

def train() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = try_cuda_device()
    LOG.info(f"Bat dau chay. Thiet bi: {device}")

    train_df, val_df = load_train_val_frames(Path(cfg.train_path), cfg.val_ratio, cfg.seed)
    model, tokenizer = build_model_and_tokenizer(cfg.model_name)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        LOG.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    train_ds = ReviewAspectDataset(train_df["_text"].tolist(), tokenizer, cfg.max_length, train_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64))
    val_ds = ReviewAspectDataset(val_df["_text"].tolist(), tokenizer, cfg.max_length, val_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

    y_train = train_df[list(ASPECT_COLS)].to_numpy(dtype=np.int64)
    pos_counts = (y_train > 0).sum(axis=0)
    neg_counts = (y_train == 0).sum(axis=0)
    pos_weights = torch.tensor(neg_counts / np.clip(pos_counts, 1, None), dtype=torch.float).to(device)
    
    criterion_aspect = [nn.BCEWithLogitsLoss(pos_weight=pos_weights[a]) for a in range(6)]
    criterion_sentiment = nn.SmoothL1Loss(beta=1.0)

    # Khởi tạo tối ưu hóa với toàn bộ tham số ban đầu
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
        optimizer=optim,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps
    )
    best_score, patience = -1.0, cfg.early_stopping_patience

    for epoch in range(1, cfg.num_epochs + 1):
        # for p in actual_model.encoder.parameters(): 
        #     p.requires_grad = False
        # if epoch <= 5:
        #     # Dành hẳn 5 epoch để các Heads (Classifier) làm quen với dữ liệu và hội tụ sơ bộ.
        #     status_log = "FREEZE ALL ENCODER (Heads Training Only)"
        #     pass
        # elif epoch <= 10:
        #     # Epoch 6 - 10: Chỉ mở khóa 4 tầng cuối (Layers 9-12)
        #     # Tập trung tinh chỉnh các tầng ngữ nghĩa cao cấp trước.
        #     status_log = "UNFREEZE LAYERS 9-12"
        #     for i in range(8, 12):
        #         for p in actual_model.encoder.encoder.layer[i].parameters(): p.requires_grad = True
        # elif epoch <= 15:
        #     status_log = "UNFREEZE LAYERS 5-12"
        #     for i in range(4, 12):
        #         for p in actual_model.encoder.encoder.layer[i].parameters(): p.requires_grad = True
        # else:
        #     # Epoch 16 trở đi: Mở khóa TOÀN BỘ PhoBERT để fine-tune đồng bộ
        #     status_log = "UNFREEZE ALL ENCODER (Full Fine-Tuning)"
        #     for p in actual_model.encoder.parameters(): p.requires_grad = True
        # for p in actual_model.encoder.parameters(): 
        #     p.requires_grad = False
        # if epoch <= 2:
        #     # Heads (Classifier) làm quen với dữ liệu và hội tụ sơ bộ.
        #     status_log = "FREEZE ALL ENCODER (Heads Training Only)"
        #     pass
        # elif epoch <= 4:
        #     # Chỉ mở khóa 4 tầng cuối (Layers 9-12)
        #     # Tập trung tinh chỉnh các tầng ngữ nghĩa cao cấp trước.
        #     status_log = "UNFREEZE LAYERS 9-12"
        #     for i in range(8, 12):
        #         for p in actual_model.encoder.encoder.layer[i].parameters(): p.requires_grad = True
        # elif epoch <= 6:
        #     status_log = "UNFREEZE LAYERS 5-12"
        #     for i in range(4, 12):
        #         for p in actual_model.encoder.encoder.layer[i].parameters(): p.requires_grad = True
        # else:
        #     # Mở khóa TOÀN BỘ PhoBERT để fine-tune đồng bộ
        #     status_log = "UNFREEZE ALL ENCODER (Full Fine-Tuning)"
        #     for p in actual_model.encoder.parameters(): p.requires_grad = True
        
        # LOG.info(f"\n--- Epoch {epoch} Strategy: {status_log} ---")

        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}/{cfg.num_epochs}",
            file=sys.stdout,  
            leave=True        
        )

        for step, batch in enumerate(pbar):
            labels = batch["labels"].to(device)
            aspect_logits, sentiment_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            
            aspect_targets = (labels > 0).float()
            sentiment_targets = labels.float() 
            
            loss_aspect = 0.0
            loss_sentiment = 0.0
            valid_sentiment_heads = 0 
            
            for a in range(6):
                loss_aspect += criterion_aspect[a](aspect_logits[:, a], aspect_targets[:, a])
                valid_mask = labels[:, a] > 0 
                
                if valid_mask.any():
                    preds_valid = sentiment_logits[valid_mask, a]
                    targets_valid = sentiment_targets[valid_mask, a]
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

        metrics = evaluate_epoch(model, val_loader, device)
        LOG.info(
            f"--> Epoch {epoch} Xong! Train Loss: {epoch_loss/len(train_loader):.4f} | "
            f"F1-Micro: {metrics['micro_f1']:.4f} | "
            f"Sentiment Quality: {metrics['sentiment_accuracy']:.4f} | "
            f"OVERALL SCORE: {metrics['overall_score']:.4f}"
        )

        if metrics["overall_score"] > best_score:
            best_score, patience = metrics["overall_score"], cfg.early_stopping_patience
            LOG.info(f"   [!] Cap nhat Best Overall: {best_score:.4f}")
            
            os.makedirs(cfg.output_dir, exist_ok=True)
            save_path = os.path.join(cfg.output_dir, "best_model_1.pth")
            
            torch.save(actual_model.state_dict(), save_path)
            LOG.info(f"   [+] Saved best weights to {save_path}")
        else:
            patience -= 1
            if patience <= 0:
                LOG.info("Dừng sớm do model không cải thiện!")
                break

if __name__ == "__main__":
    train()