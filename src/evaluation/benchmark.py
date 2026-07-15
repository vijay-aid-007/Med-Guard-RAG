import json
from pathlib import Path
from src.core.logging_config import logger
from src.core.pipeline       import run_pipeline





def _matches(model_answer: str, correct_answer: str) -> bool:
    """
    Production-grade matching — 4 tiers, no hardcoding:
    Tier 1 — exact substring (fast path)
    Tier 2 — fuzzy string match (handles typos, British/American spelling)
    Tier 3 — all key words present (handles word order variations)
    Tier 4 — semantic similarity (handles paraphrases)
    """
    import re
    from rapidfuzz import fuzz

    def clean(text: str) -> str:
        text = text.replace("**", "").replace("*", "")
        # Normalize all unicode hyphens to standard hyphen
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
        # Remove hyphens within words (hyper-kalaemia → hyperkalaemia)
        text = re.sub(r"(?<=\w)-(?=\w)", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    cleaned_answer  = clean(model_answer)
    cleaned_correct = clean(correct_answer)

    # ── Tier 1 — exact substring ──────────────────────────────────────
    if cleaned_correct in cleaned_answer:
        logger.debug(f"Tier1 match: '{cleaned_correct}'")
        return True

    # ── Tier 2 — fuzzy match ──────────────────────────────────────────
    # Searches for correct_answer as a substring anywhere in model_answer
    # partial_ratio handles length differences (short expected in long answer)
    fuzzy_score = fuzz.partial_ratio(cleaned_correct, cleaned_answer)
    logger.debug(f"Fuzzy score: {fuzzy_score} for '{cleaned_correct}'")
    if fuzzy_score >= 85:
        return True

    # ── Tier 3 — all key words present ───────────────────────────────
    # Filters out short words (articles, prepositions)
    key_words = [w for w in cleaned_correct.split() if len(w) > 3]
    if key_words and all(w in cleaned_answer for w in key_words):
        logger.debug(f"Tier3 keyword match: {key_words}")
        return True

    # ── Tier 4 — semantic similarity ─────────────────────────────────
    try:
        from src.ingestion.embedder import embed_texts
        import numpy as np
        vecs       = embed_texts([model_answer, correct_answer])
        similarity = float(np.dot(vecs[0], vecs[1]))
        logger.debug(f"Semantic similarity: {similarity:.3f} for '{correct_answer[:40]}'")
        return similarity >= 0.72
    except Exception:
        return False



def run_benchmark(
    path: str = "data/processed/benchmark_set.jsonl"
) -> dict:
    p = Path(path)
    if not p.exists():
        logger.error(f"{p} not found")
        return {}

    questions = [json.loads(line) for line in open(p, encoding="utf-8")]
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
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    with open(out / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Benchmark: {summary}")
    return summary


if __name__ == "__main__":
    run_benchmark()