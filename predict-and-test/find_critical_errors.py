#!/usr/bin/env python3
"""
Script to extract critical polar errors: Ground Truth is 5 Stars but Ensemble predicts 1 Star.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Cấu hình các hằng số giống hệ thống của em
ASPECT_COLS = ("giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam")
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 16
OUTPUT_DIR = "../models"
TEST_PATH = "../data/split_datasets/test_split.csv"
MODELS = [1]

class SimpleTextDataset(Dataset):
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

# Định nghĩa lại kiến trúc để nạp trọng số
class SharedEncoderMultiClassModel(torch.nn.Module):
    def __init__(self, model_name: str, num_aspects: int = 6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.aspect_heads = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden, 256), torch.nn.LayerNorm(256), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 1))
            for _ in range(num_aspects)
        ])
        self.sentiment_heads = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden, 256), torch.nn.LayerNorm(256), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 5))
            for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  
        aspect_logits = torch.stack([h(pooled).squeeze(-1) for h in self.aspect_heads], dim=1)
        sentiment_logits = torch.stack([h(pooled) for h in self.sentiment_heads], dim=1)
        return aspect_logits, sentiment_logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # 1. Đọc file test và chuẩn hóa text
    df_test = pd.read_csv(TEST_PATH, encoding="utf-8")
    text_col = None
    for cand in ("review", "Review", "text", "content", "comment"):
        if cand in df_test.columns:
            text_col = cand
            break
    
    texts = df_test[text_col].astype(str).str.replace("\r\n", "\n").str.strip().replace("", " ").tolist()
    y_true = df_test[list(ASPECT_COLS)].to_numpy(dtype=np.int64)

    # 2. Khởi tạo DataLoader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = SimpleTextDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Nạp 5 mô hình Fold để lấy dự đoán Ensemble
    models = []
    for f in MODELS:
        path = os.path.join(OUTPUT_DIR, f"best_model_fold_{f}.pth")
        if os.path.exists(path):
            m = SharedEncoderMultiClassModel(MODEL_NAME)
            m.load_state_dict(torch.load(path, map_location=device))
            m.to(device).eval()
            models.append(m)
            
    if not models:
        print("❌ Lỗi: Không tìm thấy file model checkpoint nào trong thư mục!")
        return

    # 4. Dự đoán Ensemble tích hợp xác suất mềm
    sum_sentiment_probs = None
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ensemble Processing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            batch_sent_probs = 0.0
            for m in models:
                _, sentiment_logits = m(input_ids, attention_mask)
                batch_sent_probs += torch.softmax(sentiment_logits, dim=-1).cpu().numpy()
                
            batch_sent_probs /= len(models)
            if sum_sentiment_probs is None:
                sum_sentiment_probs = batch_sent_probs
            else:
                sum_sentiment_probs = np.concatenate([sum_sentiment_probs, batch_sent_probs], axis=0)

    # Lấy nhãn dự đoán cảm xúc tối ưu (từ 1 đến 5)
    y_pred_sentiment = np.argmax(sum_sentiment_probs, axis=-1) # Shape: (N, 6)

    # 5. Thuật toán lọc tìm ca lỗi: Thực tế = 5 và Dự đoán = 1
    print("\n" + "="*80)
    print("🚨 DANH SÁCH CÁC CÂU REVIEW GÂY LỖI: THỰC TẾ 5 SAO - DỰ ĐOÁN 1 SAO")
    print("="*80)
    
    critical_errors_found = 0
    
    # Duyệt qua từng dòng dữ liệu (mẫu văn bản)
    for idx in range(len(df_test)):
        row_true = y_true[idx]
        row_pred = y_pred_sentiment[idx]
        
        # Duyệt qua 6 khía cạnh của dòng này
        for aspect_idx, aspect_name in enumerate(ASPECT_COLS):
            true_star = row_true[aspect_idx]
            pred_star = row_pred[aspect_idx]
            
            # Chỉ xét khi khía cạnh đó thực sự tồn tại (True > 0)
            if true_star == 5 and pred_star == 1:
                critical_errors_found += 1
                print(f"📌 [Ca lỗi thứ {critical_errors_found}]")
                print(f"   • Vị trí dòng trong file test: Dòng {idx + 2} (tính cả header)")
                print(f"   • Khía cạnh bị lỗi logic: '{aspect_name}'")
                print(f"   • Văn bản review gốc: \n     \"{texts[idx]}\"")
                print("-" * 80)
                
    if critical_errors_found == 0:
        print("🎉 Tuyệt vời! Không tìm thấy ca lỗi phân cực nghiêm trọng nào (5 sao đoán thành 1 sao).")
    else:
        print(f"➔ Tổng cộng tìm thấy {critical_errors_found} trường hợp mâu thuẫn cảm xúc nặng.")

if __name__ == "__main__":
    main()