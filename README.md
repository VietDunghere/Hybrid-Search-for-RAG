# Tài liệu dự án RAG Hybrid Search cho Văn bản Pháp luật

Dự án này xây dựng một hệ thống RAG (Retrieval-Augmented Generation) để tra cứu và trả lời câu hỏi dựa trên các văn bản pháp luật, sử dụng phương pháp tìm kiếm lai (Hybrid Search) kết hợp giữa Dense Retrievel (Vector) và Sparse Retrieval (BM25).

## 1. Cấu trúc và Mục đích các file

### `main.py`
- **Mục đích**: Là file chạy chính (entry point) của toàn bộ chương trình.
- **Chức năng**:
  - Khởi tạo RAG Pipeline.
  - Chạy pipeline với bộ dữ liệu đánh giá (`eval-thue.json`).
  - Ghi nhận kết quả trả lời và ngữ cảnh.
  - Gọi module đánh giá (RAGAS) để chấm điểm hệ thống.
- **Cách dùng**: Chạy file này để thực hiện toàn bộ quy trình từ đầu đến cuối.

### `config.py`
- **Mục đích**: Lưu trữ toàn bộ các tham số cấu hình của dự án.
- **Nội dung**: Bao gồm API Keys, tên model (LLM, Embedding), đường dẫn thư mục dữ liệu, các tham số cho thuật toán tìm kiếm (top_k, weights).

### `chunking.py`
- **Mục đích**: Chia nhỏ văn bản pháp luật thành các đoạn (chunks) có ý nghĩa.
- **Phương pháp**: **Regex-based Splitting**.
  - Sử dụng Regular Expressions (biểu thức chính quy) để nhận diện cấu trúc đặc thù của văn bản pháp luật Việt Nam (Ví dụ: "Điều 1.", "Chương I", "Phần 2",...).
  - Văn bản được chia theo các mục/điều khoản thay vì chia theo số lượng ký tự cố định, giúp giữ trọn vẹn ngữ nghĩa của quy định pháp luật.

### `embedding.py`
- **Mục đích**: Tạo vector đặc trưng (embeddings) cho văn bản và quản lý cơ sở dữ liệu vector.
- **Phương pháp**: 
  - Sử dụng thư viện `sentence-transformers`.
  - Lưu trữ vector vào **ChromaDB** (một Vector Database mã nguồn mở).
  - Embeddings được chuẩn hóa (normalize) để tính toán độ tương đồng cosine.

### `hybrid_search.py`
- **Mục đích**: Thực hiện cơ chế tìm kiếm lai (Hybrid Search).
- **Phương pháp**:
  - **Sparse Retrieval**: Sử dụng thuật toán **BM25** (thư viện `rank_bm25`) để tìm kiếm dựa trên từ khóa chính xác. Có tích hợp bộ tokenizer đơn giản cho tiếng Việt (loại bỏ stop words, ký tự đặc biệt).
  - **Dense Retrieval**: Sử dụng Vector Search để tìm kiếm dựa trên ngữ nghĩa.
  - **Fusion (Kết hợp)**: Hỗ trợ 2 phương pháp kết hợp kết quả:
    1. **Weighted Sum**: Tính tổng điểm có trọng số (VD: 0.5 * Vector_Score + 0.5 * BM25_Score).
    2. **RRF (Reciprocal Rank Fusion)**: Kết hợp dựa trên thứ hạng của kết quả trong từng danh sách tìm kiếm, không phụ thuộc vào điểm số thô.

### `rag_pipeline.py`
- **Mục đích**: Lớp điều phối chính (Orchestrator).
- **Chức năng**: Kết nối các module lại với nhau: Đọc dữ liệu -> Chunking -> Embedding -> Indexing -> Searching -> Generation (tạo câu trả lời từ LLM).

### `evaluate.py`
- **Mục đích**: Đánh giá chất lượng của hệ thống RAG.
- **Phương pháp**: Sử dụng framework **RAGAS**.
- **Metrics (Chỉ số đánh giá)**:
  - **Faithfulness**: Độ trung thực của câu trả lời so với ngữ cảnh được cung cấp.
  - **Answer Relevancy**: Độ liên quan của câu trả lời với câu hỏi.
  - **Context Precision**: Độ chính xác của các ngữ cảnh tìm được (có chứa thông tin đúng hay không).
  - **Context Recall**: Khả năng tìm kiếm được đầy đủ các thông tin cần thiết.
- **Lưu ý**: Module này sử dụng **NVIDIA NIM (Llama 3.1 70B)** đóng vai trò là "Giám khảo" (Judge LLM) để chấm điểm.

---

## 2. Hướng dẫn sử dụng

### Yêu cầu cài đặt
Trước khi chạy, hãy đảm bảo đã cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Cấu hình
Mở file `config.py` và cập nhật các thông tin quan trọng:
- `LLM_API_KEY`: API Key của NVIDIA NIM (hoặc OpenAI nếu bạn đổi base_url).
- `DATA_DIR`: Đường dẫn đến thư mục chứa file văn bản (`Data/`).
- `EVAL_FILE`: Đường dẫn đến file đánh giá (`Evaluation/eval-thue.json`).

### Chạy chương trình
Để chạy toàn bộ quy trình (Index dữ liệu -> Tìm kiếm -> Trả lời -> Đánh giá), bạn chỉ cần chạy lệnh sau trong terminal:

```bash
python main.py
```

### Quy trình chạy thực tế của `main.py`:
1. **Load Data**: Đọc các file `.txt` từ thư mục `Data`.
2. **Indexing**: 
   - Nếu `force_reindex=True`: Hệ thống sẽ thực hiện chunking và embedding lại từ đầu.
   - Nếu `force_reindex=False`: Hệ thống sẽ load index đã lưu trong ChromaDB.
3. **Retrieval & Generation**:
   - Hệ thống đọc các câu hỏi từ file `eval-thue.json`.
   - Với mỗi câu hỏi, hệ thống tìm kiếm (Hybrid Search) các đoạn văn bản liên quan.
   - Gửi câu hỏi + ngữ cảnh cho LLM để sinh câu trả lời.
4. **Evaluation**:
   - Kết quả trả lời được đưa qua RAGAS.
   - RAGAS tính toán các điểm số (metrics).
   - Kết quả cuối cùng được lưu vào thư mục `output/` dưới dạng file JSON và CSV.
