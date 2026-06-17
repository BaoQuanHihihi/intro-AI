#!/usr/bin/env python3
"""
Phân chia ngẫu nhiên tập dữ liệu gốc thành 2 tập Train (80%) và Test (20%).
Được thiết lập fixed seed để đảm bảo tính tái lập (reproducibility).
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    INPUT_PATH = "./train_boosted_cleaned.csv"
    OUTPUT_DIR = "./split_datasets"
    
    # Tạo thư mục chứa file đầu ra nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("⏳ Đang tải tập dữ liệu gốc...")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {INPUT_PATH}")
        
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    total_samples = len(df)
    print(f"📊 Tổng số mẫu hiện có: {total_samples}")

    # Thực hiện phân chia ngẫu nhiên theo tỷ lệ 80/20
    print("🔀 Đang tiến hành phân chia ngẫu nhiên (Tỷ lệ 80/20)...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        shuffle=True
    )

    train_output_path = os.path.join(OUTPUT_DIR, "train_split.csv")
    test_output_path = os.path.join(OUTPUT_DIR, "test_split.csv")

    # Ghi ra file CSV
    print("💾 Đang xuất dữ liệu ra các file CSV mới...")
    train_df.to_csv(train_output_path, index=False, encoding="utf-8")
    test_df.to_csv(test_output_path, index=False, encoding="utf-8")

    # Báo cáo kiểm tra số lượng mẫu
    print("\n=================== 🎉 HOÀN THÀNH PHÂN CHIA ===================")
    print(f"📁 Thư mục lưu trữ: {OUTPUT_DIR}")
    print(f"📝 Tập TRAIN (80%): {len(train_df)} mẫu  -> Lưu tại: {train_output_path}")
    print(f"📝 Tập TEST  (20%): {len(test_df)} mẫu  -> Lưu tại: {test_output_path}")
    print(f"📊 Tỷ lệ thực tế: {len(train_df)/total_samples*100:.1f}% Train / {len(test_df)/total_samples*100:.1f}% Test")
    print("===============================================================")

if __name__ == "__main__":
    split_dataset()