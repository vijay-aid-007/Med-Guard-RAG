"""
One sentence job: Checks the LLM's raw answer for toxicity, overconfident diagnosis language, 
and whether it's actually grounded in the retrieved context.

"""


import re
from sentence_transformers import SentenceTransformer, util
from src.core.config import settings
from src.core.logging_config import logger

TOXIC = [
    r"\b(kill yourself|kys)\b",
    r"\byou (deserve|should) (to )?die\b",
]
OVERCONFIDENT = [
    r"\byou (definitely|certainly) have\b",
    r"\bthis confirms you have\b",
    r"\byou are (definitely|certainly) (diagnosed|suffering from)\b",
]


class OutputGuardrail:

    def __init__(self):
        self.embed_model = SentenceTransformer(settings.embedding_model)

    def _is_toxic(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(p, low) for p in TOXIC)

    def _is_overconfident(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(p, low) for p in OVERCONFIDENT)

    def _grounding_score(
        self, answer: str, context_chunks: list[str]
    ) -> float:
        if not context_chunks:
            return 0.0
        combined = " ".join(context_chunks)
        a_vec = self.embed_model.encode(
            answer, convert_to_tensor=True, normalize_embeddings=True
        )
        c_vec = self.embed_model.encode(
            combined, convert_to_tensor=True, normalize_embeddings=True
        )
        return float(util.cos_sim(a_vec, c_vec)[0][0])

    def check(self, answer: str, context_chunks: list[str]) -> dict:
        if self._is_toxic(answer):
            logger.warning("Output blocked — toxic content")
            return {"passed": False, "reason": "toxic_output",
                    "grounding_score": None}

        if self._is_overconfident(answer):
            logger.warning("Output blocked — overconfident diagnosis")
            return {"passed": False, "reason": "overconfident_diagnosis",
                    "grounding_score": None}

        score = self._grounding_score(answer, context_chunks)
        if score < settings.faithfulness_threshold:
            logger.warning(f"Output blocked — low grounding ({score:.3f})")
            return {"passed": False, "reason": "low_grounding",
                    "grounding_score": score}

        return {"passed": True, "reason": None, "grounding_score": score}


_instance: OutputGuardrail | None = None


def get_output_guardrail() -> OutputGuardrail:
    global _instance
    if _instance is None:
        _instance = OutputGuardrail()
    return _instance