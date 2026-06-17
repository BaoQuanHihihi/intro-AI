import pandas as pd

# Đọc 3 file dữ liệu
df_majority = pd.read_csv("./targeted_data/majority_safe.csv")
df_minority_orig = pd.read_csv("./targeted_data/need_augmentation.csv")
df_minority_aug = pd.read_csv("./targeted_data/augmented_records.csv")

# Gộp lại thành một tập duy nhất
df_final_train = pd.concat([df_majority, df_minority_orig, df_minority_aug], ignore_index=True)

# Ép các cột nhãn về kiểu số nguyên, xử lý các dòng lỗi (nếu có)
ASPECT_COLS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]
for col in ASPECT_COLS:
    # Nếu có dòng nào bị lệch cột thành chuỗi, ép về NaN và điền số 0
    df_final_train[col] = pd.to_numeric(df_final_train[col], errors='coerce').fillna(0).astype(int)

# Tráo đổi ngẫu nhiên các dòng dữ liệu để mô hình học đều, không bị học theo cụm
df_final_train = df_final_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Lưu ra file train cuối cùng để nộp cho mô hình
df_final_train.to_csv("./train_boosted.csv", index=False, encoding="utf-8")

print(f"🚀 Đã tạo xong tập Train mới với {len(df_final_train):,} dòng dữ liệu sạch sẽ!")