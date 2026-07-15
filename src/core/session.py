import json
from typing import Any

import redis

from src.core.config         import settings

_redis_client: redis.Redis | None = None  # type: ignore[type-arg]


def get_redis_client() -> redis.Redis:  # type: ignore[type-arg]
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.redis_url, decode_responses=True
        )
    return _redis_client


class SessionManager:

    def __init__(self) -> None:
        self.redis = get_redis_client()

    def _key(self, session_id: str) -> str:
        return f"medguard:session:{session_id}"

    def get_state(self, session_id: str) -> dict[str, Any]:
        raw: Any = self.redis.get(self._key(session_id))
        if raw is None:
            return {
                "frustration_score": 0.0,
                "question_history":  [],
            }
        return json.loads(raw)  # type: ignore[arg-type]

    def update_state(
        self,
        session_id:        str,
        frustration_delta: int,
        new_question:      str,
    ) -> dict[str, Any]:
        state: dict[str, Any] = self.get_state(session_id)
        state["frustration_score"] = (
            float(state["frustration_score"]) + frustration_delta
        )
        history: list[str] = state["question_history"]
        history.append(new_question)
        state["question_history"] = history[-10:]

        self.redis.set(
            self._key(session_id),
            json.dumps(state),
            ex=settings.session_ttl_seconds,
        )
        return state

    def count_repeats(self, session_id: str, question: str) -> int:
        state:   dict[str, Any] = self.get_state(session_id)
        history: list[str]      = state["question_history"]
        normal  = question.strip().lower()
        return sum(
            1 for q in history
            if q.strip().lower() == normal
        )


_instance: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _instance
    if _instance is None:
        _instance = SessionManager()
    return _instance