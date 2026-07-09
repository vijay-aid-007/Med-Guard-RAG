import json
import pandas as pd
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_community.llms       import Ollama          # type: ignore[import-untyped]
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-untyped]

from src.core.config               import settings
from src.core.logging_config       import logger
from src.retrieval.retriever       import get_retriever
from src.retrieval.reranker        import get_reranker
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client     import get_llm_client


def build_eval_dataset(test_questions: list[dict]) -> Dataset:
    retriever = get_retriever()
    reranker  = get_reranker()
    llm       = get_llm_client()
    records: list[dict[str, Any]] = []

    for item in test_questions:
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")

        candidates = retriever.retrieve(question)
        top_chunks = reranker.rerank(question, candidates)
        contexts   = [c["text"] for c in top_chunks]
        prompt     = build_prompt(question, top_chunks)
        answer     = llm.generate(prompt)

        records.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })
        logger.info(f"Eval record built: {question[:60]}")

    return Dataset.from_list(records)


def run_evaluation(
    test_path: str = "data/processed/eval_set.jsonl",
) -> dict:
    # Lazy import — avoids module-level langchain_core.pydantic_v1 crash
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    path = Path(test_path)
    if not path.exists():
        logger.error(f"{path} not found — create eval_set.jsonl first")
        return {}

    questions = [
        json.loads(line)
        for line in open(path, encoding="utf-8")
    ]
    logger.info(f"Running RAGAS on {len(questions)} questions")

    dataset = build_eval_dataset(questions)

    judge_llm = Ollama(  # type: ignore[call-arg]
        model    = settings.llm_model,
        base_url = settings.ollama_base_url,
    )
    judge_emb = HuggingFaceEmbeddings(
        model_name = settings.embedding_model
    )

    result = evaluate(
        dataset,
        metrics    = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm        = judge_llm,
        embeddings = judge_emb,
    )

    df: pd.DataFrame = result.to_pandas()  # type: ignore[assignment]
    out_dir = Path("data/processed")
    df.to_csv(out_dir / "ragas_results.csv", index=False)

    summary = {
        "faithfulness":      float(df["faithfulness"].mean()),
        "answer_relevancy":  float(df["answer_relevancy"].mean()),
        "context_precision": float(df["context_precision"].mean()),
        "context_recall":    float(df["context_recall"].mean()),
    }

    with open(out_dir / "ragas_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"RAGAS summary: {summary}")
    return summary


if __name__ == "__main__":
    run_evaluation()