"""
Module embedding: tạo vector embeddings cho các chunks sử dụng sentence-transformers
và lưu trữ trong ChromaDB
"""
import os
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_DEVICE, CHROMA_DIR, COLLECTION_NAME


class EmbeddingManager:
    """Quản lý việc tạo embeddings và lưu trữ trong ChromaDB"""
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        print(f"🔄 Đang tải mô hình embedding: {model_name}...")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ Đã tải mô hình embedding thành công (device={device})")
        
        # Khởi tạo ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection_name = collection_name
        self.collection = None
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Tạo embeddings cho danh sách texts
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True  # Normalize cho cosine similarity
        )
        return embeddings.tolist()
    
    def encode_query(self, query: str) -> List[float]:
        """Tạo embedding cho một query đơn"""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return embedding[0].tolist()
    
    def create_collection(self, force_recreate: bool = False):
        """
        Tạo hoặc lấy collection trong ChromaDB
        """
        if force_recreate:
            try:
                self.chroma_client.delete_collection(self.collection_name)
                print(f"🗑️ Đã xóa collection cũ: {self.collection_name}")
            except Exception:
                pass
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Sử dụng cosine similarity
        )
        print(f"✅ Collection '{self.collection_name}' đã sẵn sàng (count={self.collection.count()})")
        return self.collection
    
    def index_chunks(self, chunks: List[Dict], force_recreate: bool = False):
        """
        Tạo embeddings và index tất cả chunks vào ChromaDB
        
        Args:
            chunks: Danh sách chunks từ module chunking
            force_recreate: Nếu True, xóa collection cũ và tạo lại
        """
        self.create_collection(force_recreate=force_recreate)
        
        # Nếu đã có dữ liệu và không force_recreate
        if self.collection.count() > 0 and not force_recreate:
            print(f"ℹ️ Collection đã có {self.collection.count()} chunks, bỏ qua indexing.")
            return
        
        print(f"\n🔄 Đang tạo embeddings cho {len(chunks)} chunks...")
        
        # Chuẩn bị dữ liệu
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Tạo embeddings
        embeddings = self.encode(documents)
        
        # Index vào ChromaDB theo batch (ChromaDB giới hạn ~41666 items/batch)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end]
            )
            print(f"  Indexed batch {i//batch_size + 1}: chunks {i}-{end-1}")
        
        print(f"✅ Đã index {len(chunks)} chunks vào ChromaDB")
    
    def dense_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Tìm kiếm dense (vector similarity) trong ChromaDB
        
        Returns:
            List[Dict]: Kết quả với score, content, metadata
        """
        if self.collection is None:
            self.create_collection()
        
        query_embedding = self.encode_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "dense_score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })
        
        return search_results


if __name__ == "__main__":
    from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from data_processing import load_documents
    from chunking import chunk_documents
    
    # 1. Load documents
    docs = load_documents(DATA_DIR)
    
    # 2. Chunk documents
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # 3. Create embeddings and index
    emb_manager = EmbeddingManager()
    emb_manager.index_chunks(chunks, force_recreate=True)
    
    # 4. Test dense search
    query = "Thuế giá trị gia tăng đầu ra trong dự toán xây lắp"
    results = emb_manager.dense_search(query, top_k=3)
    
    print(f"\n🔍 Dense search results for: '{query}'")
    for r in results:
        print(f"\n  [{r['chunk_id']}] score={r['dense_score']:.4f}")
        print(f"  Source: {r['metadata']['filename']}")
        print(f"  Content: {r['content'][:200]}...")
