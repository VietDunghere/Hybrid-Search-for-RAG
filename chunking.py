"""
Module chunking: chia văn bản thành các đoạn nhỏ (chunks) 
với hỗ trợ overlap và nhận diện cấu trúc văn bản pháp luật
"""
import re
from typing import List, Dict


def split_into_sections(text: str) -> List[Dict[str, str]]:
    """
    Chia văn bản theo các mục/điều/khoản/phần lớn
    Nhận diện các tiêu đề section trong văn bản pháp luật Việt Nam
    """
    # Pattern nhận diện các đầu mục trong văn bản pháp luật
    section_patterns = [
        r'^(Điều\s+\d+[\.\:])',              # Điều 1. / Điều 1:
        r'^([IVXLC]+[\.\-]\s)',               # I. / II- / III.
        r'^(\d+[\.\)]\s)',                     # 1. / 2) 
        r'^(\d+\.\d+[\.\-]\s)',               # 1.1. / 2.3-
        r'^(Chương\s+[IVXLC\d]+)',            # Chương I / Chương 1
        r'^(Phần\s+[IVXLC\d]+)',             # Phần I / Phần 1
        r'^(Mục\s+[IVXLC\d]+)',              # Mục I / Mục 1
        r'^(PHỤ LỤC)',                        # PHỤ LỤC
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in section_patterns)
    sections = []
    current_section = ""
    current_header = ""
    
    for line in text.split('\n'):
        stripped = line.strip()
        is_header = False
        
        for pattern in section_patterns:
            if re.match(pattern, stripped):
                # Lưu section hiện tại
                if current_section.strip():
                    sections.append({
                        "header": current_header,
                        "content": current_section.strip()
                    })
                current_header = stripped
                current_section = line + '\n'
                is_header = True
                break
        
        if not is_header:
            current_section += line + '\n'
    
    # Lưu section cuối cùng
    if current_section.strip():
        sections.append({
            "header": current_header,
            "content": current_section.strip()
        })
    
    return sections


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 100) -> List[str]:
    """
    Chia văn bản thành các chunk theo kích thước cố định với overlap.
    Ưu tiên cắt tại các ranh giới câu/đoạn.
    
    Args:
        text: Văn bản cần chia
        chunk_size: Kích thước tối đa mỗi chunk (theo ký tự)
        chunk_overlap: Số ký tự overlap giữa các chunk
    
    Returns:
        List[str]: Danh sách các chunk
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    # Chia theo câu (dùng dấu chấm, dấu chấm phẩy, xuống dòng)
    sentences = re.split(r'(?<=[.;:!\?\n])\s+', text)
    
    current_chunk = ""
    
    for sentence in sentences:
        # Nếu thêm câu hiện tại vẫn trong giới hạn chunk_size
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Lưu chunk hiện tại
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Nếu câu đơn lẻ dài hơn chunk_size, cắt cứng
            if len(sentence) > chunk_size:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk = current_chunk + " " + word if current_chunk else word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
            else:
                # Tạo overlap: lấy phần cuối của chunk trước
                if chunks:
                    overlap_text = chunks[-1][-chunk_overlap:] if len(chunks[-1]) > chunk_overlap else chunks[-1]
                    # Cắt overlap tại ranh giới từ
                    space_idx = overlap_text.find(' ')
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx + 1:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
    
    # Lưu chunk cuối cùng
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int = 512, chunk_overlap: int = 100) -> List[Dict]:
    """
    Chia tất cả documents thành chunks, giữ lại metadata.
    
    Args:
        documents: Danh sách documents từ data_processing
        chunk_size: Kích thước mỗi chunk
        chunk_overlap: Overlap giữa các chunk
    
    Returns:
        List[Dict]: Danh sách chunks, mỗi chunk chứa:
            - chunk_id: ID duy nhất
            - content: nội dung chunk
            - metadata: metadata từ document gốc + thông tin chunk
    """
    all_chunks = []
    chunk_id = 0
    
    for doc in documents:
        doc_content = doc["content"]
        doc_metadata = doc["metadata"]
        
        # Chia theo sections trước
        sections = split_into_sections(doc_content)
        
        for section in sections:
            section_content = section["content"]
            section_header = section["header"]
            
            # Chia mỗi section thành chunks
            text_chunks = chunk_text(section_content, chunk_size, chunk_overlap)
            
            for i, chunk_content in enumerate(text_chunks):
                chunk_metadata = {
                    **doc_metadata,
                    "section_header": section_header,
                    "chunk_index": i,
                    "total_chunks_in_section": len(text_chunks),
                }
                
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "content": chunk_content,
                    "metadata": chunk_metadata
                })
                chunk_id += 1
    
    print(f"✅ Đã tạo {len(all_chunks)} chunks từ {len(documents)} documents")
    return all_chunks


if __name__ == "__main__":
    from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from data_processing import load_documents
    
    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    
    print(f"\nTổng số chunks: {len(chunks)}")
    print("\n--- Ví dụ 3 chunks đầu tiên ---")
    for chunk in chunks[:3]:
        print(f"\n[{chunk['chunk_id']}] (from: {chunk['metadata']['filename']})")
        print(f"  Section: {chunk['metadata']['section_header'][:60]}...")
        print(f"  Content ({len(chunk['content'])} chars): {chunk['content'][:150]}...")
