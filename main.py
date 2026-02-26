"""
Main entry point: Chạy toàn bộ pipeline RAG Hybrid Search
- Xử lý dữ liệu → Chunking → Embedding → Indexing
- Hybrid Search (BM25 + Dense)
- Trả lời câu hỏi bằng LLM (NVIDIA NIM - Llama 3.1 70B)
- Đánh giá bằng RAGAS
"""
import json
import os
import sys
import time

from config import EVAL_FILE, OUTPUT_DIR
from rag_pipeline import RAGPipeline
from evaluate import run_ragas_evaluation, save_evaluation_report


def main():
    """Chạy toàn bộ pipeline"""
    start_time = time.time()
    
    print("\n" + "🔥" * 40)
    print("   RAG HYBRID SEARCH PIPELINE")
    print("   LLM: NVIDIA NIM - Llama 3.1 70B Instruct")
    print("🔥" * 40)
    
    # ============================================================
    # BƯỚC 1: Khởi tạo RAG Pipeline
    # ============================================================
    print("\n\n" + "=" * 60)
    print("📦 BƯỚC 1: KHỞI TẠO PIPELINE")
    print("=" * 60)
    
    pipeline = RAGPipeline(
        force_reindex=True,  # Set False nếu đã index rồi
        search_method="weighted",  # "weighted" hoặc "rrf"
    )
    
    # ============================================================
    # BƯỚC 2: Chạy pipeline trên evaluation queries
    # ============================================================
    print("\n\n" + "=" * 60)
    print("🏃 BƯỚC 2: CHẠY PIPELINE TRÊN EVALUATION QUERIES")
    print("=" * 60)
    
    pipeline_results = pipeline.run_evaluation(EVAL_FILE)
    
    # ============================================================
    # BƯỚC 3: In kết quả đầy đủ
    # ============================================================
    print("\n\n" + "=" * 60)
    print("📋 BƯỚC 3: KẾT QUẢ CHI TIẾT")
    print("=" * 60)
    
    formatted_output = pipeline.format_results(pipeline_results)
    print(formatted_output)
    
    # Lưu kết quả formatted
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "pipeline_output.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_output)
    print(f"\n💾 Kết quả đã lưu: {output_path}")
    
    # ============================================================
    # BƯỚC 4: Đánh giá bằng RAGAS
    # ============================================================
    print("\n\n" + "=" * 60)
    print("📊 BƯỚC 4: ĐÁNH GIÁ BẰNG RAGAS")
    print("=" * 60)
    
    ragas_scores = run_ragas_evaluation(pipeline_results)
    
    # ============================================================
    # BƯỚC 5: Lưu báo cáo
    # ============================================================
    print("\n\n" + "=" * 60)
    print("💾 BƯỚC 5: LƯU BÁO CÁO")
    print("=" * 60)
    
    save_evaluation_report(pipeline_results, ragas_scores)
    
    # Lưu RAGAS scores riêng
    scores_path = os.path.join(OUTPUT_DIR, "ragas_scores.json")
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(ragas_scores, f, ensure_ascii=False, indent=2)
    print(f"💾 RAGAS scores: {scores_path}")
    
    # ============================================================
    # TỔNG KẾT
    # ============================================================
    elapsed = time.time() - start_time
    print("\n\n" + "🎉" * 40)
    print(f"\n✅ HOÀN THÀNH! Thời gian: {elapsed:.1f}s ({elapsed/60:.1f} phút)")
    print(f"\n📁 Kết quả đã lưu tại: {OUTPUT_DIR}/")
    print(f"   - pipeline_output.txt    (kết quả chi tiết)")
    print(f"   - evaluation_details.json (chi tiết JSON)")
    print(f"   - evaluation_report.txt  (báo cáo tổng hợp)")
    print(f"   - ragas_scores.json      (điểm RAGAS)")
    
    if ragas_scores:
        print(f"\n📊 RAGAS Scores:")
        for metric, score in ragas_scores.items():
            if score is not None:
                print(f"   {metric}: {score:.4f}")
    
    print("\n" + "🎉" * 40)


def quick_test():
    """Test nhanh với 1 câu hỏi (không chạy RAGAS)"""
    print("\n🧪 QUICK TEST MODE")
    print("=" * 60)
    
    pipeline = RAGPipeline(force_reindex=True)
    
    # Test query
    query = "Theo Thông tư 01/1999/TT-BXD, giá trị dự toán xây lắp sau thuế bao gồm những thành phần nào?"
    
    print(f"\n❓ Query: {query}\n")
    result = pipeline.answer(query)
    
    print(f"🔍 Retrieved {len(result['retrieved_chunks'])} chunks:")
    for i, c in enumerate(result['retrieved_chunks']):
        print(f"\n  [{i+1}] {c['chunk_id']} (hybrid={c['hybrid_score']:.4f})")
        print(f"      Source: {c['metadata']['filename']}")
        print(f"      Content: {c['content'][:200]}...")
    
    print(f"\n🤖 LLM Answer:")
    print(result['answer'])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()
