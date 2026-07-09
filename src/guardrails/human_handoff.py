import re
from src.core.config import settings
from src.core.logging_config import logger

NEGATIVE = [
    r"\bwrong\b", r"\bnot helpful\b", r"\bbad answer\b",
    r"\buseless\b", r"\bdoesn'?t (help|make sense)\b",
    r"\bstill (confused|don'?t understand)\b",
]
POSITIVE = [
    r"\bthanks?\b", r"\bthat helps?\b", r"\bgot it\b",
    r"\bmakes sense\b", r"\bperfect\b", r"\bgreat\b",
]
HANDOFF_REQUEST = [
    r"\btalk to (a )?human\b",
    r"\bconnect me to (an? )?agent\b",
    r"\bspeak (to|with) (a )?(doctor|person)\b",
    r"\breal (doctor|person)\b",
]


class SatisfactionTracker:

    def score_turn(self, user_message: str) -> int:
        low = user_message.lower()
        if any(re.search(p, low) for p in NEGATIVE):
            return -1
        if any(re.search(p, low) for p in POSITIVE):
            return +1
        return 0

    def is_explicit_handoff_request(self, user_message: str) -> bool:
        low = user_message.lower()
        return any(re.search(p, low) for p in HANDOFF_REQUEST)

    def should_handoff(
        self,
        frustration_score: float,
        repeated_question_count: int,
        explicit_request: bool,
    ) -> dict:
        if explicit_request:
            logger.info("Handoff — explicit request")
            return {"handoff": True, "trigger": "explicit_request"}

        if repeated_question_count >= settings.repetition_handoff_count:
            logger.info(f"Handoff — repetition ({repeated_question_count}x)")
            return {"handoff": True, "trigger": "repetition"}

        if frustration_score <= -settings.frustration_block_threshold:
            logger.info(f"Handoff — frustration ({frustration_score})")
            return {"handoff": True, "trigger": "frustration_threshold"}

        return {"handoff": False, "trigger": None}


_instance: SatisfactionTracker | None = None


def get_satisfaction_tracker() -> SatisfactionTracker:
    global _instance
    if _instance is None:
        _instance = SatisfactionTracker()
    return _instance