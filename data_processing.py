"""
Module xử lý dữ liệu: đọc và tiền xử lý các file văn bản pháp luật
"""
import os
import re
import glob
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Tiền xử lý văn bản:
    - Loại bỏ khoảng trắng thừa
    - Chuẩn hóa xuống dòng
    - Loại bỏ các ký tự đặc biệt không cần thiết
    """
    # Thay thế nhiều dấu xuống dòng liên tiếp bằng 2 dấu xuống dòng
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Loại bỏ khoảng trắng đầu/cuối mỗi dòng
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Loại bỏ các dòng chỉ chứa dấu *
    text = re.sub(r'\n\*+\n', '\n', text)
    text = re.sub(r'^\*+$', '', text, flags=re.MULTILINE)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'[ \t]+', ' ', text)
    # Loại bỏ dòng trống thừa
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_doc_metadata(filename: str, content: str) -> Dict:
    """
    Trích xuất metadata từ tên file và nội dung văn bản
    """
    # Lấy số hiệu từ tên file (ví dụ: 01_1999_TT-BXD -> 01/1999/TT-BXD)
    base_name = os.path.splitext(filename)[0]
    # Loại bỏ hậu tố _1, _2... nếu có
    base_name_clean = re.sub(r'_(\d+)$', '', base_name)
    doc_number = base_name_clean.replace('_', '/')

    # Trích xuất cơ quan ban hành (nằm ở đầu file)
    issuing_body = ""
    for line in content.split('\n')[:10]:
        line = line.strip()
        if line and line != "********" and "CỘNG" not in line and "Độc lập" not in line:
            issuing_body = line
            break

    return {
        "filename": filename,
        "doc_number": doc_number,
        "issuing_body": issuing_body,
        "source": filename,
    }


def load_documents(data_dir: str) -> List[Dict]:
    """
    Đọc tất cả các file .txt trong thư mục dữ liệu
    
    Returns:
        List[Dict]: Danh sách các document, mỗi document chứa:
            - content: nội dung văn bản đã được tiền xử lý
            - metadata: thông tin metadata
    """
    documents = []
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    if not txt_files:
        print(f"[WARNING] Không tìm thấy file .txt nào trong {data_dir}")
        return documents

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        print(f"  📄 Đang đọc: {filename}")
        
        try:
            # Thử đọc với UTF-8
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except UnicodeDecodeError:
            # Fallback sang UTF-8 with BOM hoặc latin-1
            try:
                with open(filepath, 'r', encoding='utf-8-sig') as f:
                    raw_content = f.read()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin-1') as f:
                    raw_content = f.read()
        
        # Tiền xử lý văn bản
        cleaned_content = clean_text(raw_content)
        
        # Trích xuất metadata
        metadata = extract_doc_metadata(filename, cleaned_content)
        
        documents.append({
            "content": cleaned_content,
            "metadata": metadata
        })
        
    print(f"\n✅ Đã đọc {len(documents)} văn bản từ thư mục {data_dir}")
    return documents


if __name__ == "__main__":
    from config import DATA_DIR
    docs = load_documents(DATA_DIR)
    for doc in docs:
        print(f"\n--- {doc['metadata']['filename']} ---")
        print(f"  Số hiệu: {doc['metadata']['doc_number']}")
        print(f"  Cơ quan: {doc['metadata']['issuing_body']}")
        print(f"  Độ dài: {len(doc['content'])} ký tự")
        print(f"  Preview: {doc['content'][:200]}...")
