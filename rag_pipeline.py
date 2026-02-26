"""
Module RAG Pipeline: kết hợp Hybrid Search với LLM (NVIDIA NIM) 
để trả lời câu hỏi dựa trên các văn bản pháp luật
"""
import json
from typing import List, Dict, Optional

from openai import OpenAI

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    RAG_PROMPT_TEMPLATE, EVAL_FILE
)
from data_processing import load_documents
from chunking import chunk_documents
from embedding import EmbeddingManager
from hybrid_search import HybridSearchEngine


class RAGPipeline:
    """
    Pipeline RAG đầu-đến-cuối:
    1. Xử lý dữ liệu → Chunking → Embedding → Indexing
    2. Hybrid Search (BM25 + Dense)
    3. Tạo câu trả lời bằng LLM (NVIDIA NIM)
    """
    
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K,
        search_method: str = "weighted",
        force_reindex: bool = False,
    ):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.search_method = search_method
        
        # Khởi tạo LLM client (NVIDIA NIM - OpenAI compatible)
        self.llm_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )
        
        # Build pipeline
        self._build_pipeline(force_reindex)
    
    def _build_pipeline(self, force_reindex: bool):
        """Xây dựng toàn bộ pipeline"""
        print("\n" + "=" * 60)
        print("🚀 KHỞI TẠO RAG PIPELINE")
        print("=" * 60)
        
        # Step 1: Load và xử lý dữ liệu
        print("\n📂 Bước 1: Đọc và xử lý dữ liệu...")
        self.documents = load_documents(self.data_dir)
        
        # Step 2: Chunking
        print("\n✂️ Bước 2: Chia chunks...")
        self.chunks = chunk_documents(
            self.documents,
            self.chunk_size,
            self.chunk_overlap
        )
        
        # Step 3: Embedding & Indexing
        print("\n🧮 Bước 3: Tạo embeddings và indexing...")
        self.embedding_manager = EmbeddingManager()
        self.embedding_manager.index_chunks(self.chunks, force_recreate=force_reindex)
        
        # Step 4: Khởi tạo Hybrid Search Engine
        print("\n🔍 Bước 4: Khởi tạo Hybrid Search Engine...")
        self.search_engine = HybridSearchEngine(
            self.chunks,
            self.embedding_manager,
        )
        
        print("\n" + "=" * 60)
        print("✅ RAG PIPELINE ĐÃ SẴN SÀNG!")
        print(f"   - Documents: {len(self.documents)}")
        print(f"   - Chunks: {len(self.chunks)}")
        print(f"   - Search method: {self.search_method}")
        print(f"   - Top-K: {self.top_k}")
        print(f"   - LLM: {LLM_MODEL}")
        print("=" * 60)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Truy vấn hybrid search
        
        Returns:
            List[Dict]: Top-K chunks liên quan nhất
        """
        k = top_k or self.top_k
        results = self.search_engine.search(query, top_k=k, method=self.search_method)
        return results
    
    def _build_context(self, retrieved_chunks: List[Dict]) -> str:
        """Xây dựng context từ các chunks đã retrieve"""
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"]["filename"]
            section = chunk["metadata"].get("section_header", "")
            content = chunk["content"]
            context_parts.append(
                f"[Đoạn {i+1}] (Nguồn: {source}, Mục: {section})\n{content}"
            )
        return "\n\n---\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Gọi LLM để tạo câu trả lời dựa trên context
        """
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý pháp luật chuyên nghiệp, trả lời chính xác dựa trên văn bản được cung cấp."},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LỖI LLM] {str(e)}"
    
    def answer(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Pipeline hoàn chỉnh: Query → Retrieve → Generate Answer
        
        Returns:
            Dict chứa:
                - query: Câu hỏi
                - retrieved_chunks: Các chunk đã retrieve
                - context: Context đã xây dựng
                - answer: Câu trả lời từ LLM
        """
        # Step 1: Retrieve
        retrieved_chunks = self.retrieve(query, top_k)
        
        # Step 2: Build context
        context = self._build_context(retrieved_chunks)
        
        # Step 3: Generate answer
        llm_answer = self.generate_answer(query, context)
        
        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "answer": llm_answer,
        }
    
    def run_evaluation(self, eval_file: str = EVAL_FILE, top_k: Optional[int] = None) -> List[Dict]:
        """
        Chạy pipeline trên tất cả câu hỏi từ file evaluation
        
        Returns:
            List[Dict]: Kết quả cho mỗi câu hỏi
        """
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        
        results = []
        total = len(eval_data)
        
        print(f"\n{'='*80}")
        print(f"📋 BẮT ĐẦU ĐÁNH GIÁ TRÊN {total} CÂU HỎI")
        print(f"{'='*80}")
        
        for idx, item in enumerate(eval_data):
            query = item["query"]
            expected_answer = item.get("expected_answer", "")
            question_type = item.get("type", "unknown")
            case_context = item.get("case", "")
            
            # Nếu có case context, thêm vào query
            full_query = query
            if case_context:
                full_query = f"Tình huống: {case_context}\n\nCâu hỏi: {query}"
            
            print(f"\n--- Câu hỏi {idx+1}/{total} [{question_type}] ---")
            print(f"Q: {query[:100]}...")
            
            # Chạy pipeline
            result = self.answer(full_query, top_k)
            
            # Thêm thông tin evaluation
            result["expected_answer"] = expected_answer
            result["question_type"] = question_type
            result["case_context"] = case_context
            result["question_index"] = idx + 1
            
            results.append(result)
            
            print(f"A: {result['answer'][:150]}...")
        
        print(f"\n{'='*80}")
        print(f"✅ ĐÃ HOÀN THÀNH ĐÁNH GIÁ {total} CÂU HỎI")
        print(f"{'='*80}")
        
        return results
    
    def format_results(self, results: List[Dict]) -> str:
        """
        Format kết quả đầu ra theo mẫu yêu cầu
        """
        output_parts = []
        
        for r in results:
            part = []
            part.append(f"{'='*80}")
            part.append(f"📌 CÂU HỎI {r['question_index']} [{r['question_type']}]")
            part.append(f"{'='*80}")
            
            # 1. Câu hỏi
            part.append(f"\n❓ CÂU HỎI:")
            part.append(r["query"])
            
            if r.get("case_context"):
                part.append(f"\n📋 TÌNH HUỐNG:")
                part.append(r["case_context"])
            
            # 2. Top K chunks đã truy vấn
            part.append(f"\n🔍 TOP {len(r['retrieved_chunks'])} CHUNKS ĐÃ TRUY VẤN:")
            for i, chunk in enumerate(r["retrieved_chunks"]):
                part.append(f"\n  --- Chunk {i+1} ---")
                part.append(f"  ID: {chunk['chunk_id']}")
                part.append(f"  Source: {chunk['metadata']['filename']}")
                part.append(f"  Section: {chunk['metadata'].get('section_header', 'N/A')[:80]}")
                part.append(f"  Hybrid Score: {chunk['hybrid_score']:.4f}")
                part.append(f"  BM25 Score: {chunk['bm25_score']:.4f}")
                part.append(f"  Dense Score: {chunk['dense_score']:.4f}")
                part.append(f"  Content: {chunk['content'][:300]}...")
            
            # 3. Câu trả lời LLM
            part.append(f"\n🤖 CÂU TRẢ LỜI CỦA LLM:")
            part.append(r["answer"])
            
            # 4. Đáp án mong đợi
            part.append(f"\n✅ ĐÁP ÁN MONG ĐỢI:")
            part.append(r["expected_answer"])
            
            part.append("")
            output_parts.append("\n".join(part))
        
        return "\n\n".join(output_parts)


if __name__ == "__main__":
    import os
    from config import OUTPUT_DIR
    
    # Khởi tạo pipeline
    pipeline = RAGPipeline(force_reindex=True)
    
    # Test với 1 câu hỏi
    query = "Theo Thông tư 01/1999/TT-BXD, giá trị dự toán xây lắp sau thuế bao gồm những thành phần nào?"
    result = pipeline.answer(query)
    
    print(f"\n{'='*80}")
    print(f"❓ Query: {result['query']}")
    print(f"\n🔍 Retrieved {len(result['retrieved_chunks'])} chunks")
    for i, c in enumerate(result['retrieved_chunks']):
        print(f"  [{i+1}] {c['chunk_id']} (score={c['hybrid_score']:.4f}) - {c['metadata']['filename']}")
    print(f"\n🤖 Answer: {result['answer']}")
