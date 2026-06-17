import re
import pandas as pd
from typing import Dict

# 1. Bộ từ điển chuẩn hóa Teencode và Từ viết tắt đặc thù ngành dịch vụ
TEENCODE_DICT: Dict[str, str] = {
    "ksan": "khách sạn", "ks": "khách sạn", "khách san": "khách sạn",
    "nv": "nhân viên", "pv": "phục vụ", "staff": "nhân viên",
    "mng": "mọi người", "mn": "mọi người",
    "ko": "không", "khum": "không", "k": "không", "koo": "không",
    "đc": "được", "dc": "được",
    "j": "gì", "gđ": "gia đình", "t": "tôi",
    "thg": "thường", "bth": "bình thường", "ncl": "nói chung là",
    "keke": "vui vẻ", "kaka": "vui vẻ", "hihi": "vui vẻ", "kuku": "vui vẻ",
    "vs": "với", "vsinh": "vệ sinh", "chưởi": "chửi", "v*i": "vãi", "vãi": "rất",
    "tuyoi": "tươi", "tuoi": "tươi", "ngòn": "ngon", "p": "phút",
}

# 2. Bộ từ điển chuyển đổi Emoji phổ biến sang từ ngữ cảm xúc tiếng Việt
EMOJI_DICT: Dict[str, str] = {
    "😆": " vui vẻ ", "😊": " hài lòng ", "😍": " tuyệt vời ", "🥰": " yêu thích ",
    "🤤": " ngon lành ", "🤩": " xuất sắc ", "👍": " tốt ", "👌": " đồng ý ",
    "😅": " bối rối ", "🤣": " buồn cười ", "😂": " buồn cười ",
    "😡": " tồi tệ tức giận ", "😠": " khó chịu bực mình ", " dở ": " không ngon ",
    "😥": " thất vọng buồn ", "😞": " thất vọng ", "😱": " kinh ngạc ",
    "🤢": " tồi tệ ", "🤮": " quá tệ ", "😷": " kém kém "
}

def clean_text(text: object) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return " "
    
    # Bước a: Đưa về viết thường
    text = text.lower()
    
    # Bước b: Xử lý ký tự xuống dòng ẩn và khoảng trắng thừa
    text = text.replace("\r\n", " ").replace("\n", " ")
    
    # Bước c: Chuyển đổi Emoji sang văn bản tiếng Việt
    for emoji, text_rep in EMOJI_DICT.items():
        text = text.replace(emoji, text_rep)
        
    # Bước d: Tách từ sơ bộ để xử lý teencode chính xác (tránh thay thế nhầm từ chứa cụm đó)
    words = text.split()
    cleaned_words = []
    for word in words:
        # Loại bỏ các dấu câu dính vào từ (vd: "ksan," -> "ksan")
        word_clean = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"\']', '', word)
        if word_clean in TEENCODE_DICT:
            cleaned_words.append(TEENCODE_DICT[word_clean])
        else:
            cleaned_words.append(word)
            
    text = " ".join(cleaned_words)
    
    # Bước e: Dọn dẹp dấu câu thừa, giữ lại dấu chấm/phẩy bám ngữ cảnh câu
    text = re.sub(r'\s+', ' ', text) # Gộp nhiều dấu cách thành 1
    
    return text.strip() if text.strip() else " "

def process_csv(input_path: str, output_path: str):
    print("--- Đang bắt đầu chuẩn hóa dữ liệu ---")
    df = pd.read_csv(input_path)
    
    # Xác định cột chứa review
    text_col = None
    for col in ["Review", "review", "text", "content"]:
        if col in df.columns:
            text_col = col
            break
            
    if not text_col:
        raise ValueError("Không tìm thấy cột chứa văn bản review!")
        
    # Áp dụng hàm làm sạch văn bản
    df[text_col] = df[text_col].apply(clean_text)
    
    # Lưu kết quả
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"--- Đã chuẩn hóa xong và lưu tại: {output_path} ---")

# Ví dụ thực thi:
process_csv("train_boosted.csv", "train_boosted_cleaned.csv")