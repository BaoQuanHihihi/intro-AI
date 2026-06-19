#!/usr/bin/env python3
"""
Real-time Inference Script: Continuous evaluation of Vietnamese service reviews.
Press Ctrl+C to exit safely.
"""

import os
import re
import sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. CẤU HÌNH CỐ ĐỊNH & TỪ ĐIỂN TIỀN XỬ LÝ
# ==========================================
ASPECT_COLS = ("giai_tri", "luu_tru", "nha_hang", "an_uong", "van_chuyen", "mua_sam")
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
MODEL_WEIGHTS_PATH = "../models/best_model_fold_1.pth"  # Thay bằng fold tốt nhất của em

TEENCODE_DICT = {
    "ksan": "khách sạn", "ks": "khách sạn", "khách san": "khách sạn",
    "nv": "nhân viên", "pv": "phục vụ", "staff": "nhân viên",
    "mng": "mọi người", "mn": "mọi người",
    "ko": "không", "khum": "không", "k": "không", "koo": "không",
    "đc": "được", "dc": "được", "j": "gì", "gđ": "gia đình", "t": "tôi",
    "thg": "thường", "bth": "bình thường", "ncl": "nói chung là",
    "vs": "với", "vsinh": "vệ sinh", "chưởi": "chửi", "v*i": "vãi", "vãi": "rất",
    "tuyoi": "tươi", "tuoi": "tươi", "ngòn": "ngon", "p": "phút",
}

EMOJI_DICT = {
    "😆": " vui vẻ ", "😊": " hài lòng ", "😍": " tuyệt vời ", "🥰": " yêu thích ",
    "🤤": " ngon lành ", "🤩": " xuất sắc ", "👍": " tốt ", "👌": " đồng ý ",
    "😅": " bối rối ", "🤣": " buồn cười ", "😂": " buồn cười ",
    "😡": " tồi tệ tức giận ", "😠": " khó chịu bực mình ", " dở ": " không ngon ",
    "😥": " thất vọng buồn ", "😞": " thất vọng ", "😱": " kinh ngạc ",
    "🤢": " tồi tệ ", "🤮": " quá tệ ", "😷": " kém kém "
}


# ==========================================
# 2. KIẾN TRÚC MÔ HÌNH (BẮT BUỘC PHẢI KHỚP VỚI KHI TRAIN)
# ==========================================
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

# ==========================================
# 3. HÀM TIỀN XỬ LÝ SẠCH VĂN BẢN (TEXT CLEANING)
# ==========================================
def preprocess_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return " "
    text = text.lower()
    text = text.replace("\r\n", " ").replace("\n", " ")
    
    # Thay emoji
    for emoji, text_rep in EMOJI_DICT.items():
        text = text.replace(emoji, text_rep)
        
    # Thay teencode và xóa dấu câu bám dính
    words = text.split()
    cleaned_words = []
    for word in words:
        word_clean = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"\']', '', word)
        if word_clean in TEENCODE_DICT:
            cleaned_words.append(TEENCODE_DICT[word_clean])
        else:
            cleaned_words.append(word)
            
    text = " ".join(cleaned_words)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ==========================================
# 4. LUỒNG THỰC THI CHÍNH
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"📦 Môi trường phần cứng thực thi: {device}")
    
    # Khởi tạo Tokenizer
    print("⏳ Đang tải bộ mã hóa Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Khởi tạo mô hình và nạp trọng số
    print(f"⏳ Đang khởi tạo mô hình mạng và nạp trọng số từ '{MODEL_WEIGHTS_PATH}'...")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ LỖI: Không tìm thấy file trọng số tại {MODEL_WEIGHTS_PATH}. Vui lòng kiểm tra lại đường dẫn!")
        return
        
    model = SharedEncoderMultiClassModel(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Hệ thống ViABSA-Tour đã sẵn sàng nhận diện thực tế!")
    print("=" * 80)
    print("💡 HƯỚNG DẪN: Nhập đoạn review tiếng Việt của bạn vào prompt bên dưới và nhấn Enter.")
    print("🛑 Nhấn tổ hợp phím [Ctrl + C] bất kỳ lúc nào để thoát khỏi chương trình.")
    print("=" * 80)

    # Vòng lặp lắng nghe Real-time
    try:
        while True:
            raw_input = input("\n📝 Nhập câu review của khách hàng: ").strip()
            
            if not raw_input:
                print("⚠️ Câu nhập rỗng, vui lòng nhập lại.")
                continue
                
            # Pha 1: Tiền xử lý sâu
            cleaned_input = preprocess_text(raw_input)
            
            # Pha 2: Vector hóa token
            enc = tokenizer(
                cleaned_input, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            # Pha 3: Suy luận (Inference) qua mạng đa nhiệm
            with torch.no_grad():
                aspect_logits, sentiment_logits = model(input_ids, attention_mask)
                
                # Tính xác suất khía cạnh và số sao dự đoán
                aspect_probs = torch.sigmoid(aspect_logits).squeeze(0).cpu().numpy()
                sentiment_preds = (torch.argmax(sentiment_logits, dim=-1).squeeze(0).cpu().numpy()) + 1

            # Pha 4: Kết xuất báo cáo phân tích
            print("\n📊 KẾT QUẢ PHÂN TÍCH (REAL-TIME REPORT):")
            print("-" * 65)
            print(f"  • Chuỗi xử lý hệ thống: \"{cleaned_input}\"")
            print("-" * 65)
            
            any_aspect_detected = False
            for idx, aspect_name in enumerate(ASPECT_COLS):
                prob = aspect_probs[idx]
                
                # Ngưỡng threshold quyết định sự tồn tại là >= 0.5 giống khi train
                if prob >= 0.5:
                    any_aspect_detected = True
                    star = sentiment_preds[idx]
                    
                    # Ánh xạ nhãn hiển thị trực quan cho doanh nghiệp
                    star_desc = {1: "Rất không hài lòng 😡", 2: "Không hài lòng 😠", 
                                 3: "Trung bình 😐", 4: "Hài lòng 😊", 5: "Rất hài lòng 😍"}[star]
                    
                    print(f"   ➔ KHÍA CẠNH: [{aspect_name:<11}] | Xác suất: {prob*100:6.2f}% ➔ Đánh giá: {star} Sao ({star_desc})")
            
            if not any_aspect_detected:
                print("   ❓ Hệ thống không nhận diện được khía cạnh liên quan nào (Tất cả nhãn đều bằng 0).")
            print("-" * 65)

    except KeyboardInterrupt:
        print("\n\n🛑 Đã dừng!")
        sys.exit(0)

if __name__ == "__main__":
    main()