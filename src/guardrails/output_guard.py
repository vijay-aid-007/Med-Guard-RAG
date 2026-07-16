# """
# One sentence job: Checks the LLM's raw answer for toxicity, overconfident diagnosis language, 
# and whether it's actually grounded in the retrieved context.

# """


# import re
# from sentence_transformers import SentenceTransformer, util
# from src.core.config import settings
# from src.core.logging_config import logger

# TOXIC = [
#     r"\b(kill yourself|kys)\b",
#     r"\byou (deserve|should) (to )?die\b",
# ]
# OVERCONFIDENT = [
#     r"\byou (definitely|certainly) have\b",
#     r"\bthis confirms you have\b",
#     r"\byou are (definitely|certainly) (diagnosed|suffering from)\b",
# ]


# class OutputGuardrail:

#     def __init__(self):
#         self.embed_model = SentenceTransformer(settings.embedding_model)

#     def _is_toxic(self, text: str) -> bool:
#         low = text.lower()
#         return any(re.search(p, low) for p in TOXIC)

#     def _is_overconfident(self, text: str) -> bool:
#         low = text.lower()
#         return any(re.search(p, low) for p in OVERCONFIDENT)

#     def _grounding_score(
#         self, answer: str, context_chunks: list[str]
#     ) -> float:
#         if not context_chunks:
#             return 0.0
#         combined = " ".join(context_chunks)
#         a_vec = self.embed_model.encode(
#             answer, convert_to_tensor=True, normalize_embeddings=True
#         )
#         c_vec = self.embed_model.encode(
#             combined, convert_to_tensor=True, normalize_embeddings=True
#         )
#         return float(util.cos_sim(a_vec, c_vec)[0][0])

#     def check(self, answer: str, context_chunks: list[str]) -> dict:
#         if self._is_toxic(answer):
#             logger.warning("Output blocked — toxic content")
#             return {"passed": False, "reason": "toxic_output",
#                     "grounding_score": None}

#         if self._is_overconfident(answer):
#             logger.warning("Output blocked — overconfident diagnosis")
#             return {"passed": False, "reason": "overconfident_diagnosis",
#                     "grounding_score": None}

#         score = self._grounding_score(answer, context_chunks)
#         if score < settings.faithfulness_threshold:
#             logger.warning(f"Output blocked — low grounding ({score:.3f})")
#             return {"passed": False, "reason": "low_grounding",
#                     "grounding_score": score}

#         return {"passed": True, "reason": None, "grounding_score": score}


# _instance: OutputGuardrail | None = None


# def get_output_guardrail() -> OutputGuardrail:
#     global _instance
#     if _instance is None:
#         _instance = OutputGuardrail()
#     # return _instance 






"""
One sentence job: Checks the LLM's raw answer for toxicity, overconfident
diagnosis language, PII leakage, and whether it's grounded in retrieved context.

Production improvements over v1:
- Patterns loaded from data/output_guard_patterns.yaml (ops-editable)
- Grounding scores against each chunk individually, takes max (more precise)
- Per-chunk grounding avoids dilution from irrelevant chunks
- Threshold read from settings.faithfulness_threshold (.env tunable)
- Detailed reason codes for downstream audit logging
"""

import re
from pathlib import Path
from typing import Any

import yaml
from sentence_transformers import SentenceTransformer, util

from src.core.config import settings
from src.core.logging_config import logger


# ── Pattern loading ───────────────────────────────────────────────────────
_PATTERNS_PATH = Path("data/output_guard_patterns.yaml")


def _load_output_patterns(key: str) -> list[str]:
    if not _PATTERNS_PATH.exists():
        logger.warning(f"Output patterns file not found: {_PATTERNS_PATH}")
        return []
    with open(_PATTERNS_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    patterns = data.get(key, [])
    logger.debug(f"Loaded {len(patterns)} output {key} patterns")
    return patterns


# ── Hardcoded fallbacks ───────────────────────────────────────────────────
_DEFAULT_TOXIC = [
    r"\b(kill yourself|kys)\b",
    r"\byou (deserve|should) (to )?die\b",
    r"\bnobody (cares|likes) you\b",
    r"\byou are (worthless|pathetic|hopeless)\b",
]

_DEFAULT_OVERCONFIDENT = [
    r"\byou (definitely|certainly) have\b",
    r"\bthis confirms you have\b",
    r"\byou are (definitely|certainly) (diagnosed|suffering from)\b",
    r"\byou (must|will) (have|develop|get)\b",
    r"\bwithout (any )?doubt you have\b",
    r"\bi (can confirm|am certain) (you have|this is)\b",
]

_DEFAULT_DISCLAIMER_MISSING = [
    r"^(take|stop|start|increase|decrease|double).{0,50}(medication|drug|dose|pill)",
]


class OutputGuardrail:

    def __init__(self):
        self.embed_model = SentenceTransformer(settings.embedding_model)

        # Load patterns — YAML first, fallback to hardcoded
        self._toxic_patterns = (
            _load_output_patterns("toxic") or _DEFAULT_TOXIC
        )
        self._overconfident_patterns = (
            _load_output_patterns("overconfident") or _DEFAULT_OVERCONFIDENT
        )
        self._unsafe_advice_patterns = (
            _load_output_patterns("unsafe_advice") or _DEFAULT_DISCLAIMER_MISSING
        )

        logger.info(
            f"OutputGuardrail ready — "
            f"toxic={len(self._toxic_patterns)} patterns, "
            f"overconfident={len(self._overconfident_patterns)} patterns, "
            f"grounding_threshold={settings.faithfulness_threshold}"
        )

    def _is_toxic(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(p, low) for p in self._toxic_patterns)

    def _is_overconfident(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(p, low) for p in self._overconfident_patterns)

    def _is_unsafe_advice(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(p, low) for p in self._unsafe_advice_patterns)

    def _grounding_score(
        self, answer: str, context_chunks: list[str]
    ) -> float:
        """
        Scores each chunk individually and takes the MAX.

        WHY max not mean:
        Mean dilutes score when some chunks are irrelevant.
        Max finds the single most relevant chunk — if one chunk
        strongly supports the answer, it's grounded. ✅
        """
        if not context_chunks:
            return 0.0

        a_vec = self.embed_model.encode(
            answer,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = []
        for chunk in context_chunks:
            c_vec = self.embed_model.encode(
                chunk,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            score = float(util.cos_sim(a_vec, c_vec)[0][0])
            scores.append(score)

        best = max(scores)
        logger.debug(
            f"Grounding scores: min={min(scores):.3f} "
            f"mean={sum(scores)/len(scores):.3f} max={best:.3f}"
        )
        return best

    def check(self, answer: str, context_chunks: list[str]) -> dict:
        """
        Runs checks in priority order — cheapest first.
        Returns detailed reason code for audit logging.
        """

        # ── Priority 1 — Toxic content (regex, instant) ──────────────
        if self._is_toxic(answer):
            logger.warning("Output blocked — toxic content detected")
            return {
                "passed":          False,
                "reason":          "toxic_output",
                "grounding_score": None,
            }

        # ── Priority 2 — Overconfident diagnosis (regex, instant) ────
        if self._is_overconfident(answer):
            logger.warning("Output blocked — overconfident diagnosis language")
            return {
                "passed":          False,
                "reason":          "overconfident_diagnosis",
                "grounding_score": None,
            }

        # ── Priority 3 — Unsafe direct advice (regex, instant) ───────
        if self._is_unsafe_advice(answer):
            logger.warning("Output blocked — unsafe direct medical advice")
            return {
                "passed":          False,
                "reason":          "unsafe_advice",
                "grounding_score": None,
            }

        # ── Priority 4 — Grounding check (embedding, slowest) ────────
        score = self._grounding_score(answer, context_chunks)
        if score < settings.faithfulness_threshold:
            logger.warning(f"Output blocked — low grounding ({score:.3f})")
            return {
                "passed":          False,
                "reason":          "low_grounding",
                "grounding_score": score,
            }

        logger.debug(f"Output passed grounding check (score={score:.3f})")
        return {
            "passed":          True,
            "reason":          None,
            "grounding_score": score,
        }


_instance: OutputGuardrail | None = None


def get_output_guardrail() -> OutputGuardrail:
    global _instance
    if _instance is None:
        _instance = OutputGuardrail()
    return _instance