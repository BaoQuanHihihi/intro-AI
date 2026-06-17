import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập giao diện biểu đồ cho đẹp và hiển thị được tiếng Việt
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans' # Tránh lỗi hiển thị ô vuông nếu có tiếng Việt

# Định nghĩa các cột khía cạnh
ASPECT_COLS = ["giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam"]

def analyze_dataset(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file tại đường dẫn: {file_path}")
        return

    # Đọc file dữ liệu
    df = pd.read_csv(file_path)
    
    # ==================================================
    # NỘI DUNG 1: Tổng số lượng review
    # ==================================================
    total_reviews = len(df)
    print("=" * 60)
    print(f"📊 BÁO CÁO PHÂN TÍCH DỮ LIỆU: {os.path.basename(file_path)}")
    print("=" * 60)
    print(f"▶️ 1. Tổng số lượng review trong tập dữ liệu: {total_reviews:,} mẫu.")
    print("-" * 60)

    # ==================================================
    # NỘI DUNG 2: Thống kê số lượng theo từng Aspect
    # ==================================================
    # Tính số lượng review đề cập tới từng aspect (giá trị nhãn > 0)
    aspect_counts = (df[ASPECT_COLS] > 0).sum().sort_values(ascending=False)
    
    print("▶️ 2. Thống kê số lượng review theo từng Aspect (Khía cạnh):")
    for aspect, count in aspect_counts.items():
        percentage = (count / total_reviews) * 100
        print(f"   - {aspect:<12}: {count:>5,} review ({percentage:.2f}%)")
    print("-" * 60)

    # ==================================================
    # NỘI DUNG 3: Thống kê số lượng theo chất lượng đánh giá (1-5 sao)
    # ==================================================
    # Gom tụ toàn bộ các nhãn từ 1 đến 5 sao của tất cả các khía cạnh lại thành một mảng phẳng
    all_ratings = df[ASPECT_COLS].to_numpy()
    valid_ratings = all_ratings[all_ratings > 0] # Lọc bỏ nhãn 0 (không liên quan)
    
    # Đếm tần suất bằng numpy
    stars, star_counts = np.unique(valid_ratings, return_counts=True)
    
    total_valid_ratings = len(valid_ratings)
    print("▶️ 3. Thống kê số lượng theo chất lượng đánh giá (Tổng hòa các khía cạnh):")
    for star, count in zip(stars, star_counts):
        percentage = (count / total_valid_ratings) * 100
        print(f"   - {star} sao: {count:>5,} lượt đánh giá ({percentage:.2f}%)")
    print("=" * 60)

    # ==================================================
    # VẼ BIỂU ĐỒ GỘP (SUBPLOTS: 1 DÒNG - 2 CỘT)
    # ==================================================
    # Khởi tạo khung hình chung rộng 18 inch, cao 6 inch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # --- Biểu đồ 1: Aspect Distribution (Nằm bên trái - ax1) ---
    sns.barplot(x=aspect_counts.index, y=aspect_counts.values, palette="viridis", ax=ax1)
    ax1.set_title("Mất cân bằng khía cạnh - Aspect Distribution\n(Tần suất xuất hiện)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Khía cạnh (Aspects)", fontsize=11)
    ax1.set_ylabel("Số lượng review đề cập", fontsize=11)
    
    # Xoay chữ nhãn cột một chút nếu tên khía cạnh dài
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15)
    
    # Hiển thị số liệu trên đầu mỗi cột của biểu đồ 1
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    # --- Biểu đồ 2: Sentiment Distribution (Nằm bên phải - ax2) ---
    sns.barplot(x=stars, y=star_counts, palette="magma", ax=ax2)
    ax2.set_title("Mất cân bằng cảm xúc - Sentiment Distribution\n(Phân phối từ 1-5 sao)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Mức độ hài lòng (Số sao)", fontsize=11)
    ax2.set_ylabel("Tổng lượt đánh giá", fontsize=11)
    
    # Hiển thị số liệu trên đầu mỗi cột của biểu đồ 2
    for p in ax2.patches:
        ax2.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    # Tối ưu khoảng cách giữa các biểu đồ và hiển thị
    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

# Cấu hình đường dẫn file để chạy thử
# FILE_PATH = "./targeted_data/test.csv"
FILE_PATH = "./train_boosted_cleaned.csv"
analyze_dataset(FILE_PATH)