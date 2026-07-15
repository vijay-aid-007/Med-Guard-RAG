"""
One sentence job: Blocks jailbreaks, harmful requests, and off-topic
queries before they ever reach retrieval or the LLM.

Production improvements over v1:
- Domain anchors loaded from YAML (data/domain_anchors.yaml) — update
  without touching code, no redeploy needed
- Jailbreak/harmful patterns loaded from YAML too — ops team can update
- Threshold read from settings — tunable via .env
- Anchor vectors cached at startup — O(1) per query
"""

import re
from pathlib import Path
from typing import Any

import yaml
from sentence_transformers import SentenceTransformer, util

from src.core.config import settings
from src.core.logging_config import logger


# ── Pattern files — load from YAML ───────────────────────────────────────
_PATTERNS_PATH = Path("data/guardrail_patterns.yaml")
_ANCHORS_PATH  = Path("data/domain_anchors.yaml")


def _load_anchors() -> list[str]:
    """
    Loads domain anchors from YAML grouped by specialty.
    Falls back to empty list if file missing — logs warning.
    Adding new medical topics = edit YAML, no code change needed.
    """
    if not _ANCHORS_PATH.exists():
        logger.warning(f"Anchors file not found: {_ANCHORS_PATH}")
        return []

    with open(_ANCHORS_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    anchors = []
    for specialty, questions in data.items():
        if isinstance(questions, list):
            anchors.extend(questions)
            logger.debug(f"Loaded {len(questions)} anchors from [{specialty}]")

    logger.info(f"Loaded {len(anchors)} domain anchors from {_ANCHORS_PATH}")
    return anchors


def _load_patterns(key: str) -> list[str]:
    """
    Loads regex patterns from YAML.
    Falls back to hardcoded defaults if file missing.
    """
    if not _PATTERNS_PATH.exists():
        return []

    with open(_PATTERNS_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    patterns = data.get(key, [])
    logger.info(f"Loaded {len(patterns)} {key} patterns")
    return patterns


# ── Hardcoded fallback patterns (used if YAML missing) ───────────────────
_DEFAULT_JAILBREAK = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"you are now (in )?(dan|developer) mode",
    r"pretend (you are|to be) (an? )?(unfiltered|uncensored|jailbroken)",
    r"disregard (your|all) (safety|guidelines|rules)",
    r"act as (if you have )?no (restrictions|filters|rules)",
    r"reveal your (instructions|prompt|system message)",
]

_DEFAULT_HARMFUL = [
    r"how (do|can) i (make|synthesize|create) (a )?(bomb|poison|drug)",
    r"how (much|many) .{0,20}(pills|tablets) (to|will) (kill|overdose)",
    r"lethal dose",
    r"how to (harm|hurt|kill) (myself|someone)",
]


class InputGuardrail:

    def __init__(self, similarity_threshold: float | None = None):
        # Threshold from settings — tunable via .env without code change
        self.threshold = similarity_threshold or settings.input_similarity_threshold

        # Load patterns — YAML first, fallback to hardcoded
        self._jailbreak_patterns = (
            _load_patterns("jailbreak") or _DEFAULT_JAILBREAK
        )
        self._harmful_patterns = (
            _load_patterns("harmful") or _DEFAULT_HARMFUL
        )

        # Load embedding model + encode anchors once at startup
        self.embed_model = SentenceTransformer(settings.embedding_model)
        anchors          = _load_anchors()

        if not anchors:
            logger.error("No domain anchors loaded — all queries will pass domain check!")

        self.anchor_vecs = self.embed_model.encode(
            anchors,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        logger.info(
            f"InputGuardrail ready — "
            f"{len(anchors)} anchors, threshold={self.threshold}"
        )

    def _jailbreak(self, text: str) -> str | None:
        low = text.lower()
        for p in self._jailbreak_patterns:
            if re.search(p, low):
                return p
        return None

    def _harmful(self, text: str) -> str | None:
        low = text.lower()
        for p in self._harmful_patterns:
            if re.search(p, low):
                return p
        return None

    def _domain_score(self, text: str) -> float:
        q_vec  = self.embed_model.encode(
            text, convert_to_tensor=True, normalize_embeddings=True
        )
        scores = util.cos_sim(q_vec, self.anchor_vecs)[0]
        return float(scores.max())

    def check(self, query: str) -> dict:
        # Priority 1 — jailbreak (cheapest, regex)
        if match := self._jailbreak(query):
            logger.warning(f"Input blocked — jailbreak: {match}")
            return {"passed": False, "reason": "jailbreak_attempt"}

        # Priority 2 — harmful (regex)
        if match := self._harmful(query):
            logger.warning(f"Input blocked — harmful: {match}")
            return {"passed": False, "reason": "harmful_request"}

        # Priority 3 — domain relevance (embedding, only if above pass)
        score = self._domain_score(query)
        if score < self.threshold:
            logger.info(f"Input blocked — off_topic (score={score:.3f})")
            return {"passed": False, "reason": "off_topic"}

        logger.info(f"Input passed (score={score:.3f})")
        return {"passed": True, "reason": None}


_instance: InputGuardrail | None = None


def get_input_guardrail() -> InputGuardrail:
    global _instance
    if _instance is None:
        _instance = InputGuardrail()
    return _instance