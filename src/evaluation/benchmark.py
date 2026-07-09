import json
from pathlib import Path
from src.core.logging_config import logger
from src.core.pipeline       import run_pipeline


def _matches(model_answer: str, correct_answer: str) -> bool:
    """
    Tier 1: exact substring match — fast
    Tier 2: semantic similarity — catches correct paraphrases
    Example: 'reduces heart rate and contractility' matches
             'Reduction of cardiac output' at ~0.82 similarity
    """
    if correct_answer.lower().strip() in model_answer.lower():
        return True

    try:
        from src.ingestion.embedder import embed_texts
        import numpy as np
        vecs       = embed_texts([model_answer, correct_answer])
        similarity = float(np.dot(vecs[0], vecs[1]))
        logger.debug(f"Semantic similarity: {similarity:.3f} for '{correct_answer[:40]}'")
        return similarity > 0.75
    except Exception:
        return False


def run_benchmark(
    path: str = "data/processed/benchmark_set.jsonl"
) -> dict:
    p = Path(path)
    if not p.exists():
        logger.error(f"{p} not found")
        return {}

    questions = [json.loads(l) for l in open(p, encoding="utf-8")]
    logger.info(f"Benchmarking {len(questions)} questions")

    correct = blocked = handoff = 0
    results = []

    for item in questions:
        question       = item["question"]
        correct_answer = item["correct_answer"]

        result     = run_pipeline(question, 0.0, 0)
        is_correct = False

        if result.status in ("blocked_input", "blocked_output"):
            blocked += 1
        else:
            if result.status == "handoff":
                handoff += 1
            is_correct = _matches(result.final_answer, correct_answer)
            if is_correct:
                correct += 1

        results.append({
            "question":       question,
            "correct_answer": correct_answer,
            "model_answer":   result.final_answer,
            "status":         result.status,
            "is_correct":     is_correct,
        })
        logger.info(f"[{'OK' if is_correct else 'MISS'}] {question[:60]}")

    total   = len(questions)
    summary = {
        "total":              total,
        "correct":            correct,
        "accuracy":           round(correct / total, 4) if total else 0.0,
        "blocked":            blocked,
        "handoff":            handoff,
        "random_baseline":    0.25,
    }

    out = Path("data/processed")
    with open(out / "benchmark_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Benchmark: {summary}")
    return summary


if __name__ == "__main__":
    run_benchmark()