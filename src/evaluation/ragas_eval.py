"""
RAGAS evaluation using local Ollama LLM + HuggingFace embeddings.
Runs fully offline — no OpenAI API key needed.

Custom evaluation — replaces RAGAS for CPU/local LLM environments.
Computes faithfulness and relevancy using simple semantic similarity
instead of LLM-as-judge JSON parsing (which phi3:mini fails at).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer, util

from src.core.config import settings
from src.core.logging_config import logger
from src.retrieval.retriever import get_retriever
from src.retrieval.reranker import get_reranker
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import get_llm_client


def _cosine(a, b) -> float:
    return float(util.cos_sim(a, b)[0][0])


def build_eval_dataset(
    test_questions: list[dict],
    embed_model: SentenceTransformer,
) -> list[dict[str, Any]]:

    retriever = get_retriever()
    reranker  = get_reranker()
    llm       = get_llm_client()
    records   = []

    for item in test_questions:
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")

        candidates = retriever.retrieve_with_hyde(question, top_k=20)
        top_chunks = reranker.rerank(question, candidates, top_k=5)
        contexts   = [c["text"] for c in top_chunks]
        prompt     = build_prompt(question, top_chunks)
        answer     = llm.generate(prompt)

        if not answer or not answer.strip():
            answer = "No answer generated."

        records.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })
        logger.info(f"Eval record built: {question[:60]}")

    return records


def compute_metrics(
    records: list[dict],
    embed_model: SentenceTransformer,
) -> pd.DataFrame:
    rows = []

    for r in records:
        q  = r["question"]
        a  = r["answer"]
        gt = r["ground_truth"]
        ctxs = r["contexts"]

        # Encode all at once
        vecs = embed_model.encode(
            [a, gt, q] + ctxs,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        v_ans, v_gt, v_q = vecs[0], vecs[1], vecs[2]
        v_ctxs = vecs[3:]

        # ── Faithfulness ──────────────────────────────────────────────
        # How well is the answer grounded in the retrieved contexts?
        faith = float(max(_cosine(v_ans, vc) for vc in v_ctxs))

        # ── Answer Relevancy ──────────────────────────────────────────
        # How relevant is the answer to the question?
        ans_rel = _cosine(v_ans, v_q)

        # ── Context Precision ─────────────────────────────────────────
        # How many retrieved contexts are relevant to the ground truth?
        ctx_scores = [_cosine(vc, v_gt) for vc in v_ctxs]
        ctx_prec   = float(np.mean([s > 0.5 for s in ctx_scores]))

        # ── Context Recall ────────────────────────────────────────────
        # Does the best context cover the ground truth?
        ctx_recall = float(max(ctx_scores))

        rows.append({
            "question":         q,
            "faithfulness":     round(faith,     3),
            "answer_relevancy": round(ans_rel,   3),
            "context_precision":round(ctx_prec,  3),
            "context_recall":   round(ctx_recall,3),
        })
        logger.info(
            f"Metrics — faith={faith:.3f} rel={ans_rel:.3f} "
            f"prec={ctx_prec:.3f} rec={ctx_recall:.3f} | {q[:40]}"
        )

    return pd.DataFrame(rows)


def run_evaluation(
    test_path: str = "data/processed/eval_set.jsonl",
) -> dict:

    path = Path(test_path)
    if not path.exists():
        logger.error(f"{path} not found")
        return {}

    questions = [json.loads(line) for line in open(path, encoding="utf-8")]
    logger.info(f"Running evaluation on {len(questions)} questions")

    # Single embed model used for all metric computation
    embed_model = SentenceTransformer(
        settings.embedding_model,
        device="cpu",
    )

    records = build_eval_dataset(questions, embed_model)
    df      = compute_metrics(records, embed_model)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ragas_results.csv", index=False)

    summary = {
        "faithfulness":      round(float(df["faithfulness"].mean()),      3),
        "answer_relevancy":  round(float(df["answer_relevancy"].mean()),  3),
        "context_precision": round(float(df["context_precision"].mean()), 3),
        "context_recall":    round(float(df["context_recall"].mean()),    3),
    }

    with open(out_dir / "ragas_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation summary: {summary}")
    return summary


if __name__ == "__main__":
    run_evaluation()