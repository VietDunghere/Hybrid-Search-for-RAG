"""
Module đánh giá: sử dụng RAGAS framework để đánh giá chất lượng RAG pipeline
Các metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
"""
import json
import os
from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
    EVAL_FILE, OUTPUT_DIR, EMBEDDING_MODEL
)


def create_ragas_llm():
    """
    Tạo LLM wrapper cho RAGAS sử dụng NVIDIA NIM (OpenAI-compatible API)
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.1,  # dùng temperature thấp cho evaluation
        max_tokens=1024,
    )
    return LangchainLLMWrapper(llm)


def create_ragas_embeddings():
    """
    Tạo Embedding wrapper cho RAGAS sử dụng NVIDIA NIM embeddings
    """
    embeddings = OpenAIEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def prepare_ragas_dataset(pipeline_results: List[Dict]) -> Dataset:
    """
    Chuyển đổi kết quả từ RAG pipeline sang format RAGAS Dataset
    
    RAGAS yêu cầu:
    - question: câu hỏi
    - answer: câu trả lời từ LLM
    - contexts: list các context chunks
    - ground_truth: đáp án mong đợi (cho context_recall)
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for result in pipeline_results:
        questions.append(result["query"])
        answers.append(result["answer"])
        
        # Lấy content từ các chunks đã retrieve
        chunk_contents = [chunk["content"] for chunk in result["retrieved_chunks"]]
        contexts.append(chunk_contents)
        
        ground_truths.append(result.get("expected_answer", ""))
    
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    
    return Dataset.from_dict(data)


def run_ragas_evaluation(pipeline_results: List[Dict]) -> Dict:
    """
    Chạy đánh giá RAGAS trên kết quả pipeline
    
    Returns:
        Dict chứa scores cho các metrics
    """
    print("\n" + "=" * 60)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ VỚI RAGAS")
    print("=" * 60)
    
    # Chuẩn bị dataset
    print("\n🔄 Đang chuẩn bị dataset cho RAGAS...")
    dataset = prepare_ragas_dataset(pipeline_results)
    print(f"✅ Dataset: {len(dataset)} samples")
    
    # Tạo LLM và Embeddings cho RAGAS
    print("🔄 Đang khởi tạo LLM và Embeddings cho RAGAS...")
    ragas_llm = create_ragas_llm()
    ragas_embeddings = create_ragas_embeddings()
    
    # Chạy evaluation
    print("🔄 Đang chạy evaluation (có thể mất vài phút)...")
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        
        print("\n" + "=" * 60)
        print("📊 KẾT QUẢ ĐÁNH GIÁ RAGAS")
        print("=" * 60)
        print(f"\n  Faithfulness:       {result['faithfulness']:.4f}")
        print(f"  Answer Relevancy:   {result['answer_relevancy']:.4f}")
        print(f"  Context Precision:  {result['context_precision']:.4f}")
        print(f"  Context Recall:     {result['context_recall']:.4f}")
        
        # Tính average score
        avg_score = (
            result['faithfulness'] + 
            result['answer_relevancy'] + 
            result['context_precision'] + 
            result['context_recall']
        ) / 4
        print(f"\n  📈 Average Score:   {avg_score:.4f}")
        print("=" * 60)
        
        return {
            "faithfulness": float(result['faithfulness']),
            "answer_relevancy": float(result['answer_relevancy']),
            "context_precision": float(result['context_precision']),
            "context_recall": float(result['context_recall']),
            "average_score": float(avg_score),
        }
        
    except Exception as e:
        print(f"\n⚠️ Lỗi khi chạy RAGAS evaluation: {str(e)}")
        print("Đang thử chạy từng metric riêng lẻ...")
        
        individual_results = {}
        for metric in metrics:
            try:
                result = evaluate(
                    dataset=dataset,
                    metrics=[metric],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                )
                metric_name = metric.name
                individual_results[metric_name] = float(result[metric_name])
                print(f"  ✅ {metric_name}: {result[metric_name]:.4f}")
            except Exception as me:
                metric_name = metric.name
                individual_results[metric_name] = None
                print(f"  ❌ {metric_name}: Lỗi - {str(me)}")
        
        # Tính average cho các metric thành công
        valid_scores = [v for v in individual_results.values() if v is not None]
        if valid_scores:
            individual_results["average_score"] = sum(valid_scores) / len(valid_scores)
        else:
            individual_results["average_score"] = None
            
        return individual_results


def save_evaluation_report(
    pipeline_results: List[Dict],
    ragas_scores: Dict,
    output_dir: str = OUTPUT_DIR,
):
    """
    Lưu báo cáo đánh giá đầy đủ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Lưu kết quả chi tiết (JSON)
    detail_path = os.path.join(output_dir, "evaluation_details.json")
    
    # Serialize-safe results
    serializable_results = []
    for r in pipeline_results:
        sr = {
            "question_index": r.get("question_index"),
            "question_type": r.get("question_type"),
            "query": r.get("query"),
            "case_context": r.get("case_context", ""),
            "answer": r.get("answer"),
            "expected_answer": r.get("expected_answer"),
            "retrieved_chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "source": c["metadata"]["filename"],
                    "section": c["metadata"].get("section_header", ""),
                    "hybrid_score": c.get("hybrid_score", 0),
                    "bm25_score": c.get("bm25_score", 0),
                    "dense_score": c.get("dense_score", 0),
                    "content": c["content"][:500],  # Cắt bớt để file không quá lớn
                }
                for c in r.get("retrieved_chunks", [])
            ]
        }
        serializable_results.append(sr)
    
    report = {
        "ragas_scores": ragas_scores,
        "total_questions": len(pipeline_results),
        "results": serializable_results,
    }
    
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Chi tiết đã lưu: {detail_path}")
    
    # 2. Lưu báo cáo text
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ RAG PIPELINE - HYBRID SEARCH\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📊 RAGAS SCORES:\n")
        for metric, score in ragas_scores.items():
            if score is not None:
                f.write(f"  {metric}: {score:.4f}\n")
            else:
                f.write(f"  {metric}: N/A\n")
        
        f.write(f"\n📋 TỔNG SỐ CÂU HỎI: {len(pipeline_results)}\n")
        
        # Chi tiết từng câu
        from rag_pipeline import RAGPipeline
        dummy_pipeline = type('obj', (object,), {'format_results': RAGPipeline.format_results})()
        formatted = RAGPipeline.format_results(None, pipeline_results)
        f.write("\n" + formatted)
    
    print(f"💾 Báo cáo đã lưu: {report_path}")
    
    return detail_path, report_path


if __name__ == "__main__":
    # Module này được gọi từ main.py
    print("Module evaluate - sử dụng từ main.py")
    print("Chạy: python main.py")
