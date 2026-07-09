""" 
: Takes the top-20 FAISS candidates and re-scores each (query, chunk) pair jointly using a cross-encoder,
 returning only the most precisely relevant top-5.
Understand before writing:
FAISS's bi-encoder embeds query and document separately and compares vectors. The cross-encoder takes 
query and document together as one input — its attention layers can directly compare query tokens against
 document tokens, which is far more precise but too slow to run against the whole corpus. Running it on
   only 20 candidates (not 6000) keeps it fast enough for the request path.

"""

from sentence_transformers import CrossEncoder
from src.core.config import settings
from src.core.logging_config import logger

from transformers import AutoTokenizer


class Reranker:
    def __init__(self):
        logger.info(f"Loading reranker: {settings.reranker_model}")
        self.model = CrossEncoder(settings.reranker_model, max_length=512)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.rerank_top_k

        if not candidates:
            return []

        pairs: list[tuple[str, str]] = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)             # type: ignore[arg-type]

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(
            candidates,
            key=lambda c: c["rerank_score"],
            reverse=True,
        )

        logger.debug(
            f"Reranked {len(candidates)} → top {top_k} "
            f"(best={reranked[0]['rerank_score']:.3f})"
        )
        return reranked[:top_k]


_instance: Reranker | None = None


def get_reranker() -> Reranker:
    global _instance
    if _instance is None:
        _instance = Reranker()
    return _instance