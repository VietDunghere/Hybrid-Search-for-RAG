"""
Module đánh giá: sử dụng RAGAS framework (v0.4.x) để đánh giá chất lượng RAG pipeline
Các metrics: Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, ContextRecall
"""
import json
import os
from typing import List, Dict

from ragas import evaluate, RunConfig, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    ContextRecall,
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
        temperature=0.1,
        max_tokens=4096,  # Tăng max_tokens để tránh LLMDidNotFinishException
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


def prepare_ragas_dataset(pipeline_results: List[Dict]) -> EvaluationDataset:
    """
    Chuyển đổi kết quả từ RAG pipeline sang EvaluationDataset (RAGAS 0.4.x)

    RAGAS 0.4.x sử dụng SingleTurnSample với các trường:
    - user_input: câu hỏi
    - response: câu trả lời từ LLM
    - retrieved_contexts: list các context chunks (List[str])
    - reference: đáp án mong đợi (ground truth)
    """
    samples = []
    for result in pipeline_results:
        # Lấy content từ các chunks đã retrieve
        chunk_contents = [chunk["content"] for chunk in result["retrieved_chunks"]]

        # Đảm bảo mỗi context là string
        chunk_contents = [str(c) for c in chunk_contents if c]

        # Đảm bảo reference không rỗng (RAGAS cần reference cho ContextRecall)
        reference = result.get("expected_answer", "")
        if not reference or not reference.strip():
            reference = "Không có đáp án tham chiếu."

        sample = SingleTurnSample(
            user_input=str(result["query"]),
            response=str(result["answer"]),
            retrieved_contexts=chunk_contents,
            reference=str(reference),
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_ragas_evaluation(pipeline_results: List[Dict]) -> Dict:
    """
    Chạy đánh giá RAGAS trên kết quả pipeline

    Returns:
        Dict chứa scores cho các metrics
    """
    print("\n" + "=" * 60)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ VỚI RAGAS (v0.4.x)")
    print("=" * 60)

    # Chuẩn bị dataset
    print("\n🔄 Đang chuẩn bị dataset cho RAGAS...")
    dataset = prepare_ragas_dataset(pipeline_results)
    print(f"✅ Dataset: {len(dataset)} samples")

    # Tạo LLM và Embeddings cho RAGAS
    print("🔄 Đang khởi tạo LLM và Embeddings cho RAGAS...")
    ragas_llm = create_ragas_llm()
    ragas_embeddings = create_ragas_embeddings()

    # Khởi tạo metrics (RAGAS 0.4.x dùng class instances, truyền llm/embeddings khi khởi tạo)
    metrics = [
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    # Cấu hình RunConfig: chạy tuần tự để tránh lỗi với NVIDIA NIM
    run_config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=5,
    )

    print("🔄 Đang chạy evaluation (có thể mất vài phút)...")

    # Chạy từng metric riêng lẻ để tránh lỗi một metric ảnh hưởng tất cả
    all_scores = {}
    for metric, name in zip(metrics, metric_names):
        print(f"\n  🔄 Đang đánh giá: {name}...")
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[metric],
                run_config=run_config,
                raise_exceptions=False,
                batch_size=1,
            )
            # Lấy score từ result - chuyển sang pandas DataFrame
            result_df = result.to_pandas()
            # Tên cột trong kết quả = metric.name (thuộc tính của metric class)
            col_name = metric.name
            scores_series = result_df[col_name]
            # Lọc NaN
            valid_scores = scores_series.dropna().tolist()
            if valid_scores:
                avg = sum(valid_scores) / len(valid_scores)
                all_scores[name] = round(avg, 4)
                print(f"  ✅ {name}: {avg:.4f}  ({len(valid_scores)}/{len(scores_series)} samples valid)")
            else:
                all_scores[name] = None
                print(f"  ⚠️ {name}: Không có kết quả hợp lệ")
        except Exception as e:
            all_scores[name] = None
            print(f"  ❌ {name}: Lỗi - {str(e)}")

    # Tính average
    valid_scores = [v for v in all_scores.values() if v is not None]
    if valid_scores:
        all_scores["average_score"] = round(sum(valid_scores) / len(valid_scores), 4)
    else:
        all_scores["average_score"] = None

    # In tổng kết
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ RAGAS")
    print("=" * 60)
    for name in metric_names:
        score = all_scores.get(name)
        if score is not None:
            print(f"  {name:25s}: {score:.4f}")
        else:
            print(f"  {name:25s}: N/A")
    avg = all_scores.get("average_score")
    if avg is not None:
        print(f"\n  📈 {'Average Score':25s}: {avg:.4f}")
    else:
        print(f"\n  📈 {'Average Score':25s}: N/A")
    print("=" * 60)

    return all_scores


def save_evaluation_report(
    pipeline_results: List[Dict],
    ragas_scores: Dict,
    output_dir: str = OUTPUT_DIR,
):
    """
    Lưu báo cáo đánh giá đầy đủ
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Lưu RAGAS scores (JSON)
    scores_path = os.path.join(output_dir, "ragas_scores.json")
    with open(scores_path, 'w', encoding='utf-8') as f:
        json.dump(ragas_scores, f, ensure_ascii=False, indent=2)
    print(f"\n💾 RAGAS scores đã lưu: {scores_path}")

    # 2. Lưu kết quả chi tiết (JSON)
    detail_path = os.path.join(output_dir, "evaluation_details.json")

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
                    "content": c["content"][:500],
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
    print(f"💾 Chi tiết đã lưu: {detail_path}")

    # 3. Lưu báo cáo text
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

        f.write(f"\n📋 TỔNG SỐ CÂU HỎI: {len(pipeline_results)}\n\n")

        for i, r in enumerate(pipeline_results):
            f.write(f"\n{'─' * 70}\n")
            f.write(f"Câu {i+1}: {r.get('query', '')}\n")
            f.write(f"Loại: {r.get('question_type', 'N/A')}\n")
            f.write(f"Trả lời:\n{r.get('answer', '')}\n")
            f.write(f"Đáp án mong đợi:\n{r.get('expected_answer', '')}\n")
            f.write(f"Số chunks: {len(r.get('retrieved_chunks', []))}\n")

    print(f"💾 Báo cáo đã lưu: {report_path}")

    # 4. Lưu output pipeline
    output_path = os.path.join(output_dir, "pipeline_output.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pipeline_results):
            f.write(f"\n{'=' * 70}\n")
            f.write(f"CÂU HỎI {i+1}: {r.get('query', '')}\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"\n📝 TRẢ LỜI:\n{r.get('answer', '')}\n")
            f.write(f"\n📚 CHUNKS ĐÃ TÌM ({len(r.get('retrieved_chunks', []))}):\n")
            for j, c in enumerate(r.get("retrieved_chunks", [])):
                f.write(f"\n  [{j+1}] Source: {c['metadata']['filename']}")
                f.write(f" | Score: {c.get('hybrid_score', 0):.4f}\n")
                f.write(f"  {c['content'][:300]}...\n")
    print(f"💾 Pipeline output đã lưu: {output_path}")
