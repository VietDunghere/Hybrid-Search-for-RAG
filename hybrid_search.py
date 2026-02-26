"""
Module Hybrid Search: kết hợp BM25 (sparse retrieval) và Dense retrieval (vector search)
sử dụng Reciprocal Rank Fusion (RRF) hoặc weighted scoring
"""
import re
import math
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from config import TOP_K, BM25_WEIGHT, DENSE_WEIGHT
from embedding import EmbeddingManager


def vietnamese_tokenize(text: str) -> List[str]:
    """
    Tokenize tiếng Việt đơn giản (word-level).
    Chuyển thường, loại bỏ ký tự đặc biệt, split theo khoảng trắng.
    """
    text = text.lower()
    # Giữ lại chữ cái tiếng Việt, số và khoảng trắng
    text = re.sub(r'[^\wàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]', ' ', text)
    tokens = text.split()
    # Loại bỏ stop words tiếng Việt cơ bản
    stop_words = {
        'và', 'của', 'cho', 'các', 'trong', 'với', 'là', 'được', 'có', 'này',
        'đã', 'từ', 'đến', 'về', 'theo', 'như', 'tại', 'do', 'để', 'khi',
        'không', 'một', 'những', 'trên', 'bởi', 'vì', 'nếu', 'thì', 'mà',
        'hay', 'hoặc', 'nhưng', 'cũng', 'đó', 'sẽ', 'đang', 'rằng',
    }
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return tokens


class BM25Index:
    """BM25 Sparse Retrieval Index"""
    
    def __init__(self, chunks: List[Dict]):
        """
        Xây dựng BM25 index từ danh sách chunks
        """
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        
        # Tokenize tất cả documents
        print("🔄 Đang xây dựng BM25 index...")
        self.tokenized_docs = [vietnamese_tokenize(c["content"]) for c in chunks]
        
        # Tạo BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"✅ BM25 index đã sẵn sàng ({len(chunks)} documents)")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Tìm kiếm BM25 cho một query
        
        Returns:
            List[Dict]: Kết quả với bm25_score, content, metadata
        """
        tokenized_query = vietnamese_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Lấy top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Chỉ lấy kết quả có score > 0
                results.append({
                    "chunk_id": self.chunks[idx]["chunk_id"],
                    "content": self.chunks[idx]["content"],
                    "metadata": self.chunks[idx]["metadata"],
                    "bm25_score": float(scores[idx]),
                })
        
        return results


class HybridSearchEngine:
    """
    Engine Hybrid Search kết hợp BM25 và Dense Retrieval
    
    Phương pháp kết hợp:
    1. Weighted Score Fusion: Kết hợp score đã normalize theo trọng số
    2. Reciprocal Rank Fusion (RRF): Kết hợp theo thứ hạng
    """
    
    def __init__(
        self,
        chunks: List[Dict],
        embedding_manager: EmbeddingManager,
        bm25_weight: float = BM25_WEIGHT,
        dense_weight: float = DENSE_WEIGHT,
    ):
        self.chunks = chunks
        self.embedding_manager = embedding_manager
        self.bm25_index = BM25Index(chunks)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # Tạo mapping chunk_id -> chunk cho tra cứu nhanh
        self.chunk_map = {c["chunk_id"]: c for c in chunks}
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-Max normalization cho scores"""
        if not scores:
            return scores
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
    
    def search_weighted(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Hybrid search sử dụng Weighted Score Fusion
        
        Kết hợp BM25 score và Dense score (đã normalize) theo trọng số
        """
        # Lấy nhiều hơn top_k từ mỗi source để có đủ candidates
        fetch_k = top_k * 3
        
        # 1. BM25 search
        bm25_results = self.bm25_index.search(query, top_k=fetch_k)
        
        # 2. Dense search
        dense_results = self.embedding_manager.dense_search(query, top_k=fetch_k)
        
        # 3. Normalize scores
        bm25_scores = {r["chunk_id"]: r["bm25_score"] for r in bm25_results}
        dense_scores = {r["chunk_id"]: r["dense_score"] for r in dense_results}
        
        # Normalize
        if bm25_scores:
            bm25_vals = list(bm25_scores.values())
            min_b, max_b = min(bm25_vals), max(bm25_vals)
            range_b = max_b - min_b if max_b != min_b else 1.0
            bm25_norm = {k: (v - min_b) / range_b for k, v in bm25_scores.items()}
        else:
            bm25_norm = {}
        
        if dense_scores:
            dense_vals = list(dense_scores.values())
            min_d, max_d = min(dense_vals), max(dense_vals)
            range_d = max_d - min_d if max_d != min_d else 1.0
            dense_norm = {k: (v - min_d) / range_d for k, v in dense_scores.items()}
        else:
            dense_norm = {}
        
        # 4. Combine scores
        all_chunk_ids = set(bm25_norm.keys()) | set(dense_norm.keys())
        combined_results = []
        
        for chunk_id in all_chunk_ids:
            bm25_s = bm25_norm.get(chunk_id, 0.0)
            dense_s = dense_norm.get(chunk_id, 0.0)
            
            hybrid_score = self.bm25_weight * bm25_s + self.dense_weight * dense_s
            
            chunk = self.chunk_map[chunk_id]
            combined_results.append({
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "bm25_score": bm25_scores.get(chunk_id, 0.0),
                "dense_score": dense_scores.get(chunk_id, 0.0),
                "bm25_norm": bm25_s,
                "dense_norm": dense_s,
                "hybrid_score": hybrid_score,
            })
        
        # 5. Sort by hybrid score
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return combined_results[:top_k]
    
    def search_rrf(self, query: str, top_k: int = TOP_K, k: int = 60) -> List[Dict]:
        """
        Hybrid search sử dụng Reciprocal Rank Fusion (RRF)
        
        RRF score = sum(1 / (k + rank_i)) cho mỗi ranking list
        k là tham số smoothing (thường = 60)
        """
        fetch_k = top_k * 3
        
        # 1. BM25 search
        bm25_results = self.bm25_index.search(query, top_k=fetch_k)
        
        # 2. Dense search
        dense_results = self.embedding_manager.dense_search(query, top_k=fetch_k)
        
        # 3. Tính RRF scores
        rrf_scores = defaultdict(float)
        
        for rank, result in enumerate(bm25_results):
            rrf_scores[result["chunk_id"]] += 1.0 / (k + rank + 1)
        
        for rank, result in enumerate(dense_results):
            rrf_scores[result["chunk_id"]] += 1.0 / (k + rank + 1)
        
        # 4. Build result list
        bm25_map = {r["chunk_id"]: r.get("bm25_score", 0) for r in bm25_results}
        dense_map = {r["chunk_id"]: r.get("dense_score", 0) for r in dense_results}
        
        combined_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            chunk = self.chunk_map[chunk_id]
            combined_results.append({
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "bm25_score": bm25_map.get(chunk_id, 0.0),
                "dense_score": dense_map.get(chunk_id, 0.0),
                "hybrid_score": rrf_score,
            })
        
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined_results[:top_k]
    
    def search(self, query: str, top_k: int = TOP_K, method: str = "weighted") -> List[Dict]:
        """
        Hybrid search với phương pháp được chọn
        
        Args:
            query: Câu truy vấn
            top_k: Số kết quả trả về
            method: "weighted" hoặc "rrf"
        """
        if method == "weighted":
            return self.search_weighted(query, top_k)
        elif method == "rrf":
            return self.search_rrf(query, top_k)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'weighted' or 'rrf'.")


if __name__ == "__main__":
    from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from data_processing import load_documents
    from chunking import chunk_documents
    
    # Pipeline
    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    
    emb_manager = EmbeddingManager()
    emb_manager.index_chunks(chunks, force_recreate=True)
    
    search_engine = HybridSearchEngine(chunks, emb_manager)
    
    # Test query
    query = "Thuế giá trị gia tăng đầu ra trong dự toán xây lắp có mục đích sử dụng là gì?"
    
    print(f"\n{'='*80}")
    print(f"🔍 HYBRID SEARCH (Weighted)")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    results = search_engine.search(query, top_k=5, method="weighted")
    for i, r in enumerate(results):
        print(f"\n--- Top {i+1} (hybrid={r['hybrid_score']:.4f}, bm25={r['bm25_score']:.4f}, dense={r['dense_score']:.4f}) ---")
        print(f"Source: {r['metadata']['filename']}")
        print(f"Content: {r['content'][:200]}...")
    
    print(f"\n{'='*80}")
    print(f"🔍 HYBRID SEARCH (RRF)")
    print(f"{'='*80}")
    
    results_rrf = search_engine.search(query, top_k=5, method="rrf")
    for i, r in enumerate(results_rrf):
        print(f"\n--- Top {i+1} (rrf={r['hybrid_score']:.6f}, bm25={r['bm25_score']:.4f}, dense={r['dense_score']:.4f}) ---")
        print(f"Source: {r['metadata']['filename']}")
        print(f"Content: {r['content'][:200]}...")
